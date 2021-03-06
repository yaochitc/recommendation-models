package io.yaochi.recommendation.model.lr

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import io.yaochi.recommendation.model.encoder.FirstOrderEncoder
import io.yaochi.recommendation.model.{RecModel, RecModelType}
import io.yaochi.recommendation.util.GradUtil

class LR(inputDim: Int)
  extends RecModel(RecModelType.BIAS_WEIGHT) {

  private val innerModel = new InternalLRModel()

  override def getMatsSize: Array[Int] = Array()

  override def getInputDim: Int = inputDim

  override def getEmbeddingDim: Int = -1

  override protected def forward(params: Map[String, Any]): Array[Float] = {
    val batchSize = params("batch_size").asInstanceOf[Int]
    val index = params("index").asInstanceOf[Array[Long]].map(_.toInt)
    val weights = params("weights").asInstanceOf[Array[Float]]
    val bias = params("bias").asInstanceOf[Array[Float]]

    innerModel.forward(batchSize, index, weights, bias)
  }

  override protected def backward(params: Map[String, Any]): Float = {
    val batchSize = params("batch_size").asInstanceOf[Int]
    val index = params("index").asInstanceOf[Array[Long]].map(_.toInt)
    val weights = params("weights").asInstanceOf[Array[Float]]
    val bias = params("bias").asInstanceOf[Array[Float]]
    val targets = params("targets").asInstanceOf[Array[Float]]

    innerModel.backward(batchSize, index, weights, bias, targets)
  }

}

private[lr] class InternalLRModel extends Serializable {
  def forward(batchSize: Int,
              index: Array[Int],
              weights: Array[Float],
              bias: Array[Float]): Array[Float] = {
    val encoder = FirstOrderEncoder(batchSize)
    val weightTable = T(Tensor.apply(weights, Array(weights.length)),
      Tensor.apply(index, Array(index.length)))
    val weightTensor = encoder.forward(weightTable)
    val biasTensor = Tensor.apply(bias, Array(bias.length))

    val outputModule = InternalLRModel.buildOutputModule()
    val inputTable = T(weightTensor, biasTensor)
    val outputTensor = outputModule.forward(inputTable)
      .toTensor[Float]
    (0 until outputTensor.nElement()).map(i => outputTensor.valueAt(i + 1, 1))
      .toArray
  }

  def backward(batchSize: Int,
               index: Array[Int],
               weights: Array[Float],
               bias: Array[Float],
               targets: Array[Float]): Float = {
    val encoder = FirstOrderEncoder(batchSize)
    val weightTable = T(Tensor.apply(weights, Array(weights.length)),
      Tensor.apply(index, Array(index.length)))
    val weightTensor = encoder.forward(weightTable)
    val biasTensor = Tensor.apply(bias, Array(bias.length))

    val inputTable = T(weightTensor, biasTensor)
    val targetTensor = Tensor.apply(targets.map(label => if (label > 0) 1.0f else 0f), Array(targets.length, 1))

    val outputModule = InternalLRModel.buildOutputModule()
    val criterion = InternalLRModel.buildCriterion()
    val outputTensor = outputModule.forward(inputTable)
    val loss = criterion.forward(outputTensor, targetTensor)
    val gradTable = outputModule.backward(inputTable, criterion.backward(outputTensor, targetTensor))
      .toTable

    val weightGradTensor = encoder.backward(weightTable, gradTable[Tensor[Float]](1))[Tensor[Float]](1)
    val biasGradTensor = gradTable[Tensor[Float]](2)

    GradUtil.weightsGrad(weights, weightGradTensor)
    GradUtil.biasGrad(bias, biasGradTensor)

    loss
  }
}

private[lr] object InternalLRModel {
  def buildCriterion() = new BCECriterion[Float]()

  def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(CAddTable())
      .add(Sigmoid[Float]())
  }
}