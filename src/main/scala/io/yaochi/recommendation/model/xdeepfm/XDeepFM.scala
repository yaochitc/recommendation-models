package io.yaochi.recommendation.model.xdeepfm

import com.intel.analytics.bigdl.nn.{BCECriterion, CAddTable, Sequential, Sigmoid}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import io.yaochi.recommendation.model.encoder.FirstOrderEncoder
import io.yaochi.recommendation.model.{RecModel, RecModelType}
import io.yaochi.recommendation.util.BackwardUtil

class XDeepFM(inputDim: Int, nFields: Int, embeddingDim: Int, fcDims: Array[Int], cinDims: Array[Int])
  extends RecModel(RecModelType.BIAS_WEIGHT_EMBEDDING_MATS) {

  private val innerModel = new InternalXDeepFMModel(nFields, embeddingDim, fcDims, cinDims)

  override def getMatsSize: Array[Int] = {
    val fcDimsArr = Array(nFields * embeddingDim) ++ fcDims
    val fcParamSize = (1 until fcDimsArr.length)
      .map(i => Array(fcDimsArr(i - 1), fcDimsArr(i), fcDimsArr(i), 1))
      .reduce(_ ++ _)

    val cinDimArr = Array(nFields) ++ cinDims
    val cinParamSize = (1 until cinDimArr.length)
      .map(i => Array(nFields * cinDimArr(i - 1), cinDimArr(i), cinDimArr(i), 1))
      .reduce(_ ++ _)

    val concatedInputDim = cinDims.sum + fcDims.last
    fcParamSize ++ cinParamSize ++ Array(concatedInputDim, 1, 1, 1)
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

private[xdeepfm] class InternalXDeepFMModel(nFields: Int,
                                            embeddingDim: Int,
                                            fcDims: Array[Int],
                                            cinDims: Array[Int]) extends Serializable {
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
    val cinEncoder = CINEncoder(batchSize, nFields, embeddingDim, fcDims, cinDims, mats)
    val cinOutputTensor = cinEncoder.forward(embeddingTensor)

    val biasTensor = Tensor.apply(bias, Array(bias.length))

    val inputTable = T.array(Array(firstOrderTensor, cinOutputTensor, biasTensor))

    val outputTensor = InternalXDeepFMModel.model.forward(inputTable).toTensor[Float]
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
    val cinEncoder = CINEncoder(batchSize, nFields, embeddingDim, fcDims, cinDims, mats)
    val cinOutputTensor = cinEncoder.forward(embeddingTensor)

    val biasTensor = Tensor.apply(bias, Array(bias.length))

    val inputTable = T.array(Array(firstOrderTensor, cinOutputTensor, biasTensor))
    val targetTensor = Tensor.apply(targets, Array(targets.length, 1))

    val model = InternalXDeepFMModel.model
    val criterion = InternalXDeepFMModel.criterion
    val outputTensor = model.forward(inputTable)
    val loss = criterion.forward(outputTensor, targetTensor)
    val gradTable = model.backward(inputTable, criterion.backward(outputTensor, targetTensor)).toTable

    val weightGradTensor = firstOrderEncoder.backward(weightTable, gradTable[Tensor[Float]](1))[Tensor[Float]](1)
    val embeddingGradTensor = cinEncoder.backward(embeddingTensor, gradTable[Tensor[Float]](2))
    val biasGradTensor = gradTable[Tensor[Float]](3)

    BackwardUtil.weightsBackward(weights, weightGradTensor)
    BackwardUtil.biasBackward(bias, biasGradTensor)
    BackwardUtil.embeddingBackward(embedding, Array(embeddingGradTensor))

    loss
  }
}

private[xdeepfm] object InternalXDeepFMModel {
  private val model = buildModel()

  private val criterion = new BCECriterion[Float]()

  def buildModel(): Sequential[Float] = {
    Sequential[Float]()
      .add(CAddTable())
      .add(Sigmoid[Float]())
  }
}