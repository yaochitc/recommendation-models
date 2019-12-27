package io.yaochi.recommendation.model.deepfm

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import io.yaochi.recommendation.model.encoder.{FirstOrderEncoder, HigherOrderEncoder, SecondOrderEncoder}
import io.yaochi.recommendation.model.{RecModel, RecModelType}
import io.yaochi.recommendation.util.BackwardUtil

class DeepFM(inputDim: Int, nFields: Int, embeddingDim: Int, fcDims: Array[Int])
  extends RecModel(RecModelType.BIAS_WEIGHT_EMBEDDING_MATS) {

  private val innerModel = new InternalDeepFMModel(nFields, embeddingDim, fcDims)

  override def getMatsSize: Array[Int] = {
    val dims = Array(nFields * embeddingDim) ++ fcDims ++ Array(1)
    (1 until dims.length)
      .map(i => Array(dims(i - 1), dims(i), dims(i), 1))
      .reduce(_ ++ _)
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

private[deepfm] class InternalDeepFMModel(nFields: Int,
                                          embeddingDim: Int,
                                          fcDims: Array[Int]) {
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
    val secondOrderEncoder = SecondOrderEncoder(batchSize, nFields, embeddingDim)
    val secondOrderTensor = secondOrderEncoder.forward(embeddingTensor)

    val biasTensor = Tensor.apply(bias, Array(bias.length))

    val higherOrderEncoder = HigherOrderEncoder(batchSize, nFields, embeddingDim, fcDims, mats)
    val higherOrderTensor = higherOrderEncoder.forward(embeddingTensor)

    val inputTable = T.array(Array(firstOrderTensor, secondOrderTensor, higherOrderTensor, biasTensor))

    val outputTensor = InternalDeepFMModel.model.forward(inputTable).toTensor[Float]
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
    val secondOrderEncoder = SecondOrderEncoder(batchSize, nFields, embeddingDim)
    val secondOrderTensor = secondOrderEncoder.forward(embeddingTensor)

    val biasTensor = Tensor.apply(bias, Array(bias.length))

    val higherOrderEncoder = HigherOrderEncoder(batchSize, nFields, embeddingDim, fcDims, mats)
    val higherOrderTensor = higherOrderEncoder.forward(embeddingTensor)

    val inputTable = T.array(Array(firstOrderTensor, secondOrderTensor, higherOrderTensor, biasTensor))
    val targetTensor = Tensor.apply(targets, Array(targets.length, 1))

    val model = InternalDeepFMModel.model
    val criterion = InternalDeepFMModel.criterion
    val outputTensor = model.forward(inputTable)
    val loss = criterion.forward(outputTensor, targetTensor)
    val gradTable = model.backward(inputTable, criterion.backward(outputTensor, targetTensor)).toTable

    val weightGradTensor = firstOrderEncoder.backward(weightTable, gradTable[Tensor[Float]](1))[Tensor[Float]](1)
    val secondOrderGradTensor = secondOrderEncoder.backward(embeddingTensor, gradTable[Tensor[Float]](2))
      .toTensor[Float]
    val higherOrderGradTensor = higherOrderEncoder.backward(embeddingTensor, gradTable[Tensor[Float]](3))
      .toTensor[Float]
    val biasGradTensor = gradTable[Tensor[Float]](4)

    BackwardUtil.weightsBackward(weights, bias, weightGradTensor, biasGradTensor)
    BackwardUtil.embeddingBackward(embedding, Array(secondOrderGradTensor, higherOrderGradTensor))

    loss
  }
}

private[deepfm] object InternalDeepFMModel {
  private val model = buildModel()

  private val criterion = new BCECriterion[Float]()

  def buildModel(): Sequential[Float] = {
    Sequential[Float]()
      .add(CAddTable())
      .add(Sigmoid[Float]())
  }
}
