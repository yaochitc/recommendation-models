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

  private val (crossLinearLayers, crossLinearEndOffset) = buildCrossLinearLayers()

  private val crossMMModules = buildCrossMMModules()

  private val (crossBiasLayers, crossBiasEndOffset) = buildCrossBiasLayers()

  private val crossOutputModules = buildCrossOutputModules()

  private val (dnnLinearLayers, dnnLinearEndOffset) = buildLinearLayers()

  private val dnnModule = buildDNNModule()

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
      val mmOutputTensor = crossMMModules(i).forward(T(x0Tensor, linearOutputTensor))
      xkTensor = crossOutputModules(i).forward(T(mmOutputTensor, xkTensor)).toTensor[Float]
    }

    crossInputTensors = inputTensors.toArray

    val dnnOutputTensor = dnnModule.forward(x0Tensor)
    outputModule.forward(T(xkTensor, dnnOutputTensor)).toTensor[Float]
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val outputModuleInput = T.apply(
      crossOutputModules(crossDepth - 1).output.toTensor[Float],
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
      val mmOutputTensor = crossMMModules(i).output.toTensor[Float]
      val addGradTable = crossOutputModules(i).backward(T(mmOutputTensor, inputTensor), lastGradTensor)
        .toTable
      xkGradTensor.add(addGradTable[Tensor[Float]](2))
      val linearOutputTensor = crossLinearLayers(i).output.toTensor[Float]
      val mmGradTable = crossMMModules(i).backward(T(x0Tensor, linearOutputTensor), addGradTable[Tensor[Float]](1))
        .toTable
      x0TensorGrad.add(mmGradTable[Tensor[Float]](1))

      xkGradTensor.add(crossLinearLayers(i).backward(inputTensor, mmGradTable[Tensor[Float]](2)))
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

  private def buildCrossMMModules(): Array[Sequential[Float]] = {
    val layers = ArrayBuffer[Sequential[Float]]()
    for (_ <- 0 until crossDepth) {
      layers += Sequential[Float]()
        .add(ParallelTable[Float]()
          .add(Unsqueeze(3))
          .add(Unsqueeze(3)))
        .add(MM[Float]())
        .add(Squeeze(3))
    }
    layers.toArray
  }

  private def buildCrossOutputModules(): Array[Sequential[Float]] = {
    val layers = ArrayBuffer[Sequential[Float]]()
    for (i <- 0 until crossDepth) {
      layers += Sequential[Float]()
        .add(CAddTable[Float]())
        .add(crossBiasLayers(i))
    }
    layers.toArray
  }

  private def buildCrossLinearLayers(): (Array[Linear[Float]], Int) = {
    val layers = ArrayBuffer[Linear[Float]]()
    var curOffset = start
    for (_ <- 0 until crossDepth) {
      layers += LayerUtil.buildLinear(xDim, 1, mats, false, curOffset)
      curOffset += xDim * 1
    }
    (layers.toArray, curOffset)
  }

  private def buildCrossBiasLayers(): (Array[CAdd[Float]], Int) = {
    val layers = ArrayBuffer[CAdd[Float]]()
    var curOffset = crossLinearEndOffset
    for (i <- 0 until crossDepth) {
      layers += LayerUtil.buildBiasLayer(1, mats, curOffset)
      curOffset += 1
    }
    (layers.toArray, curOffset)
  }

  private def buildDNNModule(): Sequential[Float] = {
    val encoder = Sequential[Float]()

    for (linearLayer <- dnnLinearLayers) {
      encoder.add(linearLayer)
      encoder.add(ReLU())
    }
    encoder
  }

  private def buildLinearLayers(): (Array[Linear[Float]], Int) = {
    val layers = ArrayBuffer[Linear[Float]]()
    var curOffset = crossBiasEndOffset
    var dim = xDim
    for (fcDim <- fcDims) {
      layers += LayerUtil.buildLinear(dim, fcDim, mats, true, curOffset)
      curOffset += dim * fcDim + fcDim
      dim = fcDim
    }
    (layers.toArray, curOffset)
  }

  private def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(JoinTable(2, 2))
      .add(outputLinearLayer)
  }

  private def buildOutputLinearLayer(): Linear[Float] = {
    val dim = xDim + fcDims.last
    LayerUtil.buildLinear(dim, 1, mats, false, dnnLinearEndOffset)
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
