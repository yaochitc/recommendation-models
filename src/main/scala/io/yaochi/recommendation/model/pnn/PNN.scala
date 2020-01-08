package io.yaochi.recommendation.model.pnn

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import io.yaochi.recommendation.model.encoder.{FirstOrderEncoder, HigherOrderEncoder}
import io.yaochi.recommendation.model.{RecModel, RecModelType}
import io.yaochi.recommendation.util.BackwardUtil

class PNN(inputDim: Int, nFields: Int, embeddingDim: Int, fcDims: Array[Int])
  extends RecModel(RecModelType.BIAS_WEIGHT_EMBEDDING_MATS) {

  private val innerModel = new InternalPNNModel(nFields, embeddingDim, fcDims)

  override def getMatsSize: Array[Int] = {
    val numPairs = nFields * (nFields - 1) / 2
    val pnnParamSize = Array(nFields * embeddingDim, fcDims.head,
      numPairs, fcDims.head)

    val fcDimsArr = fcDims ++ Array(1)
    val fcParamSize = (1 until fcDimsArr.length)
      .map(i => Array(fcDimsArr(i - 1), fcDimsArr(i), fcDimsArr(i), 1))
      .reduce(_ ++ _)
    pnnParamSize ++ fcParamSize
  }

  override def getInputDim: Int = inputDim

  override def getEmbeddingDim: Int = embeddingDim

  override protected def forward(params: Map[String, Any]): Array[Float] = {
    val batchSize = params("batch_size").asInstanceOf[Int]
    val index = params("index").asInstanceOf[Array[Long]].map(_.toInt)
    val weights = params("weights").asInstanceOf[Array[Float]]
    val bias = params("bias").asInstanceOf[Array[Float]]
    val embedding = params("embedding").asInstanceOf[Array[Float]]
    val mats = params("mats").asInstanceOf[Array[Float]]

    innerModel.forward(batchSize, index, weights, bias, embedding, mats)
  }

  override protected def backward(params: Map[String, Any]): Float = {
    val batchSize = params("batch_size").asInstanceOf[Int]
    val index = params("index").asInstanceOf[Array[Long]].map(_.toInt)
    val weights = params("weights").asInstanceOf[Array[Float]]
    val bias = params("bias").asInstanceOf[Array[Float]]
    val embedding = params("embedding").asInstanceOf[Array[Float]]
    val mats = params("mats").asInstanceOf[Array[Float]]
    val targets = params("targets").asInstanceOf[Array[Float]]

    innerModel.backward(batchSize, index, weights, bias, embedding, mats, targets)
  }

}

private[pnn] class InternalPNNModel(nFields: Int,
                                    embeddingDim: Int,
                                    fcDims: Array[Int]) extends Serializable {
  def forward(batchSize: Int,
              index: Array[Int],
              weights: Array[Float],
              bias: Array[Float],
              embedding: Array[Float],
              mats: Array[Float]): Array[Float] = {
    val weightTable = T.array(Array(Tensor.apply(weights, Array(weights.length)),
      Tensor.apply(index, Array(index.length))))

    val firstOrderEncoder = FirstOrderEncoder(batchSize)
    val firstOrderTensor = firstOrderEncoder.forward(weightTable)

    val embeddingTensor = Tensor.apply(embedding, Array(embedding.length))
    val productEncoder = ProductEncoder(batchSize, nFields, embeddingDim, fcDims.head, mats)
    val productTensor = productEncoder.forward(embeddingTensor)

    val biasTensor = Tensor.apply(bias, Array(bias.length))

    val offset = productEncoder.getParameterSize
    val dnnEncoder = DNNEncoder(batchSize, fcDims.head, fcDims.slice(1, fcDims.length), mats, offset)
    val dnnTensor = dnnEncoder.forward(productTensor)

    val inputTable = T.array(Array(firstOrderTensor, dnnTensor, biasTensor))

    val outputModule = InternalPNNModel.buildOutputModule()
    val outputTensor = outputModule.forward(inputTable).toTensor[Float]
    (0 until outputTensor.nElement()).map(i => outputTensor.valueAt(i + 1, 1))
      .toArray
  }

  def backward(batchSize: Int,
               index: Array[Int],
               weights: Array[Float],
               bias: Array[Float],
               embedding: Array[Float],
               mats: Array[Float],
               targets: Array[Float]): Float = {
    val weightTable = T.array(Array(Tensor.apply(weights, Array(weights.length)),
      Tensor.apply(index, Array(index.length))))

    val firstOrderEncoder = FirstOrderEncoder(batchSize)
    val firstOrderTensor = firstOrderEncoder.forward(weightTable)

    val embeddingTensor = Tensor.apply(embedding, Array(embedding.length))
    val productEncoder = ProductEncoder(batchSize, nFields, embeddingDim, fcDims.head, mats)
    val productTensor = productEncoder.forward(embeddingTensor)

    val biasTensor = Tensor.apply(bias, Array(bias.length))

    val offset = productEncoder.getParameterSize
    val dnnEncoder = DNNEncoder(batchSize, fcDims.head, fcDims.slice(1, fcDims.length), mats, offset)
    val dnnTensor = dnnEncoder.forward(productTensor)

    val inputTable = T.array(Array(firstOrderTensor, dnnTensor, biasTensor))
    val targetTensor = Tensor.apply(targets.map(label => if (label > 0) 1.0f else 0f), Array(targets.length, 1))

    val outputModule = InternalPNNModel.buildOutputModule()
    val criterion = InternalPNNModel.buildCriterion()
    val outputTensor = outputModule.forward(inputTable)
    val loss = criterion.forward(outputTensor, targetTensor)
    val gradTable = outputModule.backward(inputTable, criterion.backward(outputTensor, targetTensor)).toTable

    val weightGradTensor = firstOrderEncoder.backward(weightTable, gradTable[Tensor[Float]](1))[Tensor[Float]](1)
    val dnnGradTensor = dnnEncoder.backward(productTensor, gradTable[Tensor[Float]](2))
    val productGradTensor = productEncoder.backward(embeddingTensor, dnnGradTensor)
    val biasGradTensor = gradTable[Tensor[Float]](3)

    BackwardUtil.weightsBackward(weights, weightGradTensor)
    BackwardUtil.biasBackward(bias, biasGradTensor)
    BackwardUtil.embeddingBackward(embedding, Array(productGradTensor))

    loss
  }
}

private[pnn] object InternalPNNModel {
  def buildCriterion() = new BCECriterion[Float]()

  def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(CAddTable())
      .add(Sigmoid[Float]())
  }
}
