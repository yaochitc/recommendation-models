package io.yaochi.recommendation.model.dcn

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import io.yaochi.recommendation.util.{BackwardUtil, LayerUtil}

import scala.collection.mutable.ArrayBuffer

class CrossEncoder(batchSize: Int,
                   nFields: Int,
                   embeddingDim: Int,
                   crossDepth: Int,
                   fcDims: Array[Int],
                   mats: Array[Float],
                   start: Int = 0) {

  private val xDim = nFields * embeddingDim

  private val shapeModule = buildShapeModule()

  private val crossLinearLayers = buildCrossLinearLayers()

  private val crossMMLayers = buildCrossMulLayers()

  private val crossBiasOffset = start + xDim * crossDepth

  private val crossBiasLayers = buildCrossBiasLayers()

  private val crossAddLayers = buildAddLayers()

  private val dnnLinearOffset = crossBiasOffset + xDim * crossDepth

  private val dnnLinearLayers = buildLinearLayers()

  private val dnnModule = buildDNNModule()

  private val outputLinearOffset = dnnLinearOffset + getDNNParameterSize

  private val outputLinearLayer = buildOutputLinearLayer()

  private val outputModule = buildOutputModule()

  private var crossInputTensors: Array[Tensor[Float]] = _

  def forward(input: Tensor[Float]): Tensor[Float] = {
    val x0Tensor = shapeModule.forward(input)
    var xkTensor = x0Tensor
    val inputTensors = ArrayBuffer[Tensor[Float]]()
    for (i <- 0 until crossDepth) {
      inputTensors += xkTensor
      val linearOutputTensor = crossLinearLayers(i).forward(xkTensor)
      val mmOutputTensor = crossMMLayers(i).forward(T(x0Tensor, linearOutputTensor))
      xkTensor = crossAddLayers(i).forward(T(mmOutputTensor, xkTensor)).toTensor[Float]
    }

    crossInputTensors = inputTensors.toArray

    val dnnOutputTensor = dnnModule.forward(x0Tensor)
    outputModule.forward(T(xkTensor, dnnOutputTensor)).toTensor[Float]
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val outputModuleInput = T.apply(
      crossAddLayers(crossDepth - 1).output.toTensor[Float],
      dnnModule.output.toTensor[Float]
    )

    val x0Tensor = shapeModule.output.toTensor[Float]
    val x0TensorGrad = Tensor[Float]().resizeAs(x0Tensor)

    val outputModuleGradTable = outputModule.backward(outputModuleInput, gradOutput).toTable
    val dnnGradTensor = dnnModule.backward(x0Tensor, outputModuleGradTable[Tensor[Float]](2))
      .toTensor[Float]

    var lastGradTensor = outputModuleGradTable[Tensor[Float]](1)
    for (i <- crossDepth - 1 to 0 by -1) {
      val xkGradTensor = Tensor[Float]().resizeAs(x0Tensor)
      val inputTensor = crossInputTensors(i)
      val mmOutputTensor = crossMMLayers(i).output.toTensor[Float]
      val addGradTable = crossAddLayers(i).backward(T(mmOutputTensor, inputTensor), lastGradTensor)
        .toTable
      xkGradTensor.add(addGradTable[Tensor[Float]](2))
      val linearOutputTensor = crossLinearLayers(i).output.toTensor[Float]
      val mmGradTable = crossMMLayers(i).backward(T(x0Tensor, linearOutputTensor), addGradTable[Tensor[Float]](1))
        .toTable
      x0TensorGrad.add(mmGradTable[Tensor[Float]](1))

      xkGradTensor.add(crossLinearLayers(i).backward(inputTensor, mmGradTable[Tensor[Float]](2).sum(2)))
      lastGradTensor = xkGradTensor
    }

    var curOffset = start
    for (linearLayer <- crossLinearLayers) {
      curOffset = BackwardUtil.linearBackward(linearLayer, mats, curOffset)
    }

    for (biasLayer <- crossBiasLayers) {
      curOffset = BackwardUtil.biasBackward(biasLayer, mats, curOffset)
    }

    for (linearLayer <- dnnLinearLayers) {
      curOffset = BackwardUtil.linearBackward(linearLayer, mats, curOffset)
    }

    BackwardUtil.linearBackward(outputLinearLayer, mats, curOffset)

    x0TensorGrad.add(dnnGradTensor).add(lastGradTensor)

    shapeModule.backward(input, x0TensorGrad)
  }

  private def buildShapeModule(): Reshape[Float] = {
    Reshape(Array(batchSize, xDim), Some(false))
  }

  private def buildCrossMulLayers(): Array[CMulTable[Float]] = {
    val layers = ArrayBuffer[CMulTable[Float]]()
    for (_ <- 0 until crossDepth) {
      layers += CMulTable[Float]()
    }
    layers.toArray
  }

  private def buildAddLayers(): Array[Sequential[Float]] = {
    val layers = ArrayBuffer[Sequential[Float]]()
    for (i <- 0 until crossDepth) {
      layers += Sequential[Float]()
        .add(CAddTable[Float]())
        .add(crossBiasLayers(i))
    }
    layers.toArray
  }


  private def buildCrossLinearLayers(): Array[Linear[Float]] = {
    val layers = ArrayBuffer[Linear[Float]]()
    var curOffset = start
    for (i <- 0 until crossDepth) {
      layers += LayerUtil.buildLinear(xDim, 1, mats, false, curOffset)
      curOffset += xDim * 1
    }
    layers.toArray
  }

  private def buildCrossBiasLayers(): Array[CAdd[Float]] = {
    val layers = ArrayBuffer[CAdd[Float]]()
    var curOffset = crossBiasOffset
    for (i <- 0 until crossDepth) {
      layers += LayerUtil.buildBiasLayer(xDim, mats, curOffset)
      curOffset += xDim
    }
    layers.toArray
  }

  private def buildDNNModule(): Sequential[Float] = {
    val encoder = Sequential[Float]()

    for (linearLayer <- dnnLinearLayers) {
      encoder.add(linearLayer)
      encoder.add(ReLU())
    }
    encoder
  }

  private def buildLinearLayers(): Array[Linear[Float]] = {
    val layers = ArrayBuffer[Linear[Float]]()
    var curOffset = dnnLinearOffset
    var dim = xDim
    for (fcDim <- fcDims) {
      layers += LayerUtil.buildLinear(dim, fcDim, mats, true, curOffset)
      curOffset += dim * fcDim + fcDim
      dim = fcDim
    }
    layers.toArray
  }

  private def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(JoinTable(2, 2))
      .add(outputLinearLayer)
  }

  private def buildOutputLinearLayer(): Linear[Float] = {
    val dim = xDim + fcDims.last
    LayerUtil.buildLinear(dim, 1, mats, false, outputLinearOffset)
  }

  private def getDNNParameterSize: Int = {
    val dims = Array(xDim) ++ fcDims
    (1 until dims.length)
      .map(i => dims(i - 1) * dims(i) + dims(i))
      .sum
  }
}

object CrossEncoder {
  def apply(batchSize: Int,
            nFields: Int,
            embeddingDim: Int,
            crossDepth: Int,
            fcDims: Array[Int],
            mats: Array[Float],
            start: Int = 0): CrossEncoder = new CrossEncoder(batchSize, nFields, embeddingDim, crossDepth, fcDims, mats, start)
}
