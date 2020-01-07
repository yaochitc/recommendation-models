package io.yaochi.recommendation.model.xdeepfm

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{T, Table}
import io.yaochi.recommendation.util.{BackwardUtil, LayerUtil}

import scala.collection.mutable.ArrayBuffer


class CINEncoder(batchSize: Int,
                 nFields: Int,
                 embeddingDim: Int,
                 fcDims: Array[Int],
                 cinDims: Array[Int],
                 mats: Array[Float],
                 start: Int = 0) {
  private val shapeModule = buildShapeModule()

  private val dnnLinearLayers = buildLinearLayers()

  private val dnnModule = buildDNNModules()

  private val cinLinearOffset = start + getDNNParameterSize

  private val (cinLinearLayers, cinModules) = buildCINModules()

  private val sumModule = buildSumModule()

  private val outputLinearOffset = cinLinearOffset + getCINParameterSize

  private val outputLinearLayer = buildOutputLinearLayer()

  private val outputModule = buildOutputModule()

  private var cinOutputTable: Table = _

  private var cinInputTable: Table = _

  def forward(input: Tensor[Float]): Tensor[Float] = {
    val x0Tensor = shapeModule.forward(input)
    var xkTensor = x0Tensor
    var inputTable = T.apply(x0Tensor, xkTensor)
    val inputTensors = ArrayBuffer[Tensor[Float]]()
    val outputTensors = ArrayBuffer[Tensor[Float]]()
    for (cinModule <- cinModules) {
      inputTensors += xkTensor.toTensor[Float]
      xkTensor = cinModule.forward(inputTable)
      inputTable = T.apply(x0Tensor, xkTensor)
      outputTensors += xkTensor.toTensor[Float]
    }

    cinOutputTable = T.array(outputTensors.toArray)
    cinInputTable = T.array(inputTensors.toArray)

    val dinOutputTensor = sumModule.forward(cinOutputTable)

    val dnnOutputTensor = dnnModule.forward(input).toTensor[Float]

    outputModule.forward(T.apply(dinOutputTensor, dnnOutputTensor))
      .toTensor[Float]
  }

  def backward(input: Tensor[Float], gradOutput: Tensor[Float]): Tensor[Float] = {
    val outputModuleInput = T.apply(
      sumModule.output.toTensor[Float],
      dnnModule.output.toTensor[Float]
    )

    val outputModuleGradTable = outputModule.backward(outputModuleInput, gradOutput).toTable
    val dnnGradTensor = dnnModule.backward(input, outputModuleGradTable[Tensor[Float]](2))
      .toTensor[Float]

    val sumModuleGradTable = sumModule.backward(cinOutputTable, outputModuleGradTable[Tensor[Float]](1))
      .toTable

    val x0Tensor = shapeModule.output.toTensor[Float]
    val x0TensorGrad = Tensor[Float]().resizeAs(x0Tensor)

    for (i <- cinModules.length to 1 by -1) {
      var lastGradTensor = sumModuleGradTable[Tensor[Float]](i)
      for (j <- i to 1 by -1) {
        val cinInputTensor = cinInputTable[Tensor[Float]](j)
        val cinGradTable = cinModules(j - 1).backward(T.apply(x0Tensor, cinInputTensor), lastGradTensor)
          .toTable
        x0TensorGrad.add(cinGradTable[Tensor[Float]](1))
        lastGradTensor = cinGradTable[Tensor[Float]](2)
      }
      x0TensorGrad.add(lastGradTensor)
    }

    BackwardUtil.linearBackward(outputLinearLayer, mats, outputLinearOffset)

    var curOffset = start
    for (linearLayer <- dnnLinearLayers) {
      val inputSize = linearLayer.inputSize
      val outputSize = linearLayer.outputSize
      BackwardUtil.linearBackward(linearLayer, mats, curOffset)
      curOffset += inputSize * outputSize + outputSize
    }

    for (linearLayer <- cinLinearLayers) {
      val inputSize = linearLayer.inputSize
      val outputSize = linearLayer.outputSize
      BackwardUtil.linearBackward(linearLayer, mats, curOffset)
      curOffset += inputSize * outputSize + outputSize
    }

    val cinGradTensor = shapeModule.backward(input, x0TensorGrad)
      .toTensor[Float]

    cinGradTensor.add(dnnGradTensor)
  }

  private def buildShapeModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields, embeddingDim), Some(false)))
      .add(Transpose(Array((2, 3))))
  }

  private def buildDNNModules(): Sequential[Float] = {
    val encoder = Sequential[Float]()
      .add(Reshape(Array(batchSize, nFields * embeddingDim), Some(false)))

    for (linearLayer <- dnnLinearLayers) {
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
    layers.toArray
  }

  private def buildCINModules(): (Array[Linear[Float]], Array[Sequential[Float]]) = {
    val modules = ArrayBuffer[Sequential[Float]]()
    var lastCinDim = nFields
    var curOffset = cinLinearOffset
    val linearLayers = ArrayBuffer[Linear[Float]]()
    for (cinDim <- cinDims) {
      val linearLayer = LayerUtil.buildLinear(nFields * lastCinDim, cinDim, mats, true, curOffset)
      modules += buildCINModule(linearLayer, lastCinDim, cinDim, curOffset)
      linearLayers += linearLayer
      curOffset += nFields * lastCinDim * cinDim + cinDim
      lastCinDim = cinDim
    }
    (linearLayers.toArray, modules.toArray)
  }

  private def buildCINModule(linearLayer: Linear[Float], lastCinDim: Int, cinDim: Int, curOffset: Int): Sequential[Float] = {
    val x0 = Sequential[Float]()
      .add(Reshape(Array(batchSize * embeddingDim, nFields, 1), Some(false)))

    val xk = Sequential[Float]()
      .add(Reshape(Array(batchSize * embeddingDim, lastCinDim, 1), Some(false)))

    Sequential[Float]()
      .add(ParallelTable[Float]().add(x0).add(xk))
      .add(MM(transB = true))
      .add(Reshape(Array(nFields * lastCinDim)))
      .add(linearLayer)
      .add(ReLU())
      .add(Reshape(Array(batchSize, embeddingDim, cinDim), Some(false)))
  }

  private def buildSumModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(JoinTable(2, 2))
      .add(Transpose(Array((2, 3))))
      .add(Sum(dimension = 3))
  }

  private def buildOutputModule(): Sequential[Float] = {
    Sequential[Float]()
      .add(JoinTable(2, 2))
      .add(outputLinearLayer)
  }

  private def buildOutputLinearLayer(): Linear[Float] = {
    val dim = cinDims.sum + fcDims.last
    LayerUtil.buildLinear(dim, 1, mats, false, outputLinearOffset)
  }

  private def getDNNParameterSize: Int = {
    val dims = Array(nFields * embeddingDim) ++ fcDims
    (1 until dims.length)
      .map(i => dims(i - 1) * dims(i) + dims(i))
      .sum
  }

  private def getCINParameterSize: Int = {
    val dims = Array(nFields) ++ cinDims
    (1 until dims.length)
      .map(i => nFields * dims(i - 1) * dims(i) + dims(i))
      .sum
  }
}

object CINEncoder {
  def apply(batchSize: Int,
            nFields: Int,
            embeddingDim: Int,
            fcDims: Array[Int],
            cinDims: Array[Int],
            mats: Array[Float],
            start: Int = 0): CINEncoder = new CINEncoder(batchSize, nFields, embeddingDim, fcDims, cinDims, mats, start)
}

