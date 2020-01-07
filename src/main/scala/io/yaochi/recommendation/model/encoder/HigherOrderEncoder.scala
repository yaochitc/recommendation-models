package io.yaochi.recommendation.model.encoder

import com.intel.analytics.bigdl.nn.{Linear, ReLU, Reshape, Sequential}
import com.intel.analytics.bigdl.tensor.Tensor
import io.yaochi.recommendation.util.{BackwardUtil, LayerUtil}

import scala.collection.mutable.ArrayBuffer

class HigherOrderEncoder(batchSize: Int,
                         nFields: Int,
                         embeddingDim: Int,
                         fcDims: Array[Int],
                         mats: Array[Float],
                         start: Int = 0) {
  private val linearLayers = buildLinearLayers()
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

    gradTensor
  }

  private def buildModule(): Sequential[Float] = {
    val encoder = Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields * embeddingDim), Some(false)))

    for (linearLayer <- linearLayers) {
      encoder.add(linearLayer)
      encoder.add(ReLU())
    }
    encoder
  }

  private def buildLinearLayers(): Array[Linear[Float]] = {
    val layers = ArrayBuffer[Linear[Float]]()
    var curOffset = start
    var dim = nFields * embeddingDim
    for (fcDim <- fcDims) {
      layers += LayerUtil.buildLinear(dim, fcDim, mats, true, curOffset)
      curOffset += dim * fcDim + fcDim
      dim = fcDim
    }
    layers += LayerUtil.buildLinear(dim, 1, mats, true, curOffset)
    layers.toArray
  }
}

object HigherOrderEncoder {
  def apply(batchSize: Int,
            nFields: Int,
            embeddingDim: Int,
            fcDims: Array[Int],
            mats: Array[Float],
            start: Int = 0): HigherOrderEncoder =
    new HigherOrderEncoder(batchSize, nFields, embeddingDim, fcDims, mats, start)
}
