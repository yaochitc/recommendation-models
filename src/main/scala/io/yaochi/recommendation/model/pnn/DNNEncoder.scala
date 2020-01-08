package io.yaochi.recommendation.model.pnn

import com.intel.analytics.bigdl.nn.{Linear, ReLU, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import io.yaochi.recommendation.util.{BackwardUtil, LayerUtil}

import scala.collection.mutable.ArrayBuffer

class DNNEncoder(batchSize: Int,
                 inputSize: Int,
                 fcDims: Array[Int],
                 mats: Array[Float],
                 start: Int = 0) {
  private val (linearLayers, outputLinearLayer) = buildLinearLayers()
  private val module = buildModule()

  def forward(input: Tensor[Float]): Tensor[Float] = {
    module.forward(input).toTensor
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val gradTensor = module.backward(input, gradOutput).toTensor[Float]
    var curOffset = start
    for (linearLayer <- linearLayers) {
      val inputSize = linearLayer.inputSize
      val outputSize = linearLayer.outputSize

      BackwardUtil.linearBackward(linearLayer, mats, curOffset)
      curOffset += inputSize * outputSize + outputSize
    }

    BackwardUtil.linearBackward(outputLinearLayer, mats, curOffset)

    gradTensor
  }

  private def buildModule(): Sequential[Float] = {
    val encoder = Sequential[Float]()

    for (linearLayer <- linearLayers) {
      encoder.add(linearLayer)
      encoder.add(ReLU())
    }
    encoder.add(outputLinearLayer)
    encoder
  }

  private def buildLinearLayers(): (Array[Linear[Float]], Linear[Float]) = {
    val layers = ArrayBuffer[Linear[Float]]()
    var curOffset = start
    var dim = inputSize
    for (fcDim <- fcDims) {
      layers += LayerUtil.buildLinear(dim, fcDim, mats, true, curOffset)
      curOffset += dim * fcDim + fcDim
      dim = fcDim
    }
    (layers.toArray, LayerUtil.buildLinear(dim, 1, mats, true, curOffset))
  }
}

object DNNEncoder {
  def apply(batchSize: Int,
            inputSize: Int,
            fcDims: Array[Int],
            mats: Array[Float],
            start: Int = 0): DNNEncoder = new DNNEncoder(batchSize, inputSize, fcDims, mats, start)
}