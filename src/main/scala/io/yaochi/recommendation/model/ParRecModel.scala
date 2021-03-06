package io.yaochi.recommendation.model

import java.util.concurrent.Future
import java.util.{ArrayList => JArrayList}

import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix
import com.tencent.angel.ml.math2.storage.{IntFloatDenseVectorStorage, IntFloatSparseVectorStorage}
import com.tencent.angel.ml.math2.vector.{IntFloatVector, Vector}
import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.ml.matrix.psf.update.XavierUniform
import com.tencent.angel.ml.matrix.psf.update.base.VoidResult
import com.tencent.angel.ps.storage.partitioner.ColumnRangePartitioner
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.spark.models.impl.{PSMatrixImpl, PSVectorImpl}
import com.tencent.angel.spark.models.{PSMatrix, PSVector}
import io.yaochi.recommendation.data.SampleParser
import io.yaochi.recommendation.optim.AsyncOptim
import it.unimi.dsi.fastutil.ints.{Int2FloatOpenHashMap, IntOpenHashSet}

class ParRecModel(optim: AsyncOptim, model: RecModel) extends Serializable {

  var bias: PSVector = _
  var weights: PSVector = _
  var embedding: PSMatrix = _
  var mats: PSVector = _
  var params: RecModelParams = _

  val biasName = "bias"
  val weightsName = "weights"
  val embeddingName = "embedding"
  val matsName = "mats"

  val useAsync = true
  var totalPullTime: Long = 0
  var totalPushTime: Long = 0
  var totalMakeParamTime: Long = 0
  var totalCalTime: Long = 0
  var totalMakeGradTime: Long = 0
  var totalCallNum: Long = 0
  var totalWaitPullTime: Long = 0

  def init(): Unit = {
    this.params = new RecModelParams
    params.dim = model.getInputDim
    model.getType match {
      case RecModelType.BIAS_WEIGHT =>
        initMats(params.dim)

      case RecModelType.BIAS_WEIGHT_EMBEDDING =>
        params.embeddingDim = model.getEmbeddingDim
        initMats(params.dim)
        initMats(params.dim, params.embeddingDim)

      case RecModelType.BIAS_WEIGHT_EMBEDDING_MATS
           | RecModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        params.embeddingDim = model.getEmbeddingDim
        params.matSizes = model.getMatsSize
        println("matSizes is: " + params.matSizes.mkString(" "))
        initMats(params.dim)
        initMats(params.dim, params.embeddingDim)
        initMats(params.matSizes)

    }
  }

  def initEmbedding(): Unit = {
    embedding.psfUpdate(new XavierUniform(embedding.id, 0, params.embeddingDim, 1.0,
      embedding.rows, embedding.columns)).get()
  }

  def initMats(): Unit =
    mats.psfUpdate(new XavierUniform(mats.poolId, 0, 1, 1.0, 1, mats.dimension)).get()

  def initMats(inputDim: Long): Unit = {
    val biasCtx = new MatrixContext(biasName, optim.getNumSlots(), 1)
    biasCtx.setRowType(RowType.T_FLOAT_DENSE)
    biasCtx.setPartitionerClass(classOf[ColumnRangePartitioner])

    val weightCtx = new MatrixContext(weightsName, optim.getNumSlots(), inputDim)
    weightCtx.setRowType(RowType.T_FLOAT_DENSE)
    weightCtx.setPartitionerClass(classOf[ColumnRangePartitioner])

    val list = new JArrayList[MatrixContext]()
    list.add(biasCtx)
    list.add(weightCtx)

    val master = PSAgentContext.get().getMasterClient
    master.createMatrices(list, Long.MaxValue)
    bias = new PSVectorImpl(master.getMatrix(biasCtx.getName).getId, 0,
      biasCtx.getColNum, biasCtx.getRowType)
    weights = new PSVectorImpl(master.getMatrix(weightCtx.getName).getId, 0,
      weightCtx.getColNum, weightCtx.getRowType)
  }

  def initMats(inputDim: Long, embeddingDim: Int): Unit = {
    val embeddingCtx = new MatrixContext(embeddingName, embeddingDim * optim.getNumSlots(), inputDim)
    embeddingCtx.setRowType(RowType.T_FLOAT_DENSE)
    embeddingCtx.setPartitionerClass(classOf[ColumnRangePartitioner])

    val master = PSAgentContext.get().getMasterClient
    master.createMatrix(embeddingCtx, Long.MaxValue)
    embedding = new PSMatrixImpl(master.getMatrix(embeddingCtx.getName).getId, embeddingCtx.getRowNum,
      embeddingCtx.getColNum, embeddingCtx.getRowType)
    initEmbedding()
  }

  def initMats(matSizes: Array[Int]): Unit = {
    var sumDim = 0L
    var i = 0
    while (i < matSizes.length) {
      sumDim += matSizes(i) * matSizes(i + 1)
      i += 2
    }

    val matCtx = new MatrixContext(matsName, optim.getNumSlots(), sumDim)
    matCtx.setPartitionerClass(classOf[ColumnRangePartitioner])
    matCtx.setRowType(RowType.T_FLOAT_DENSE)
    val master = PSAgentContext.get().getMasterClient
    master.createMatrix(matCtx, Long.MaxValue)
    val matId = master.getMatrix(matCtx.getName).getId
    mats = new PSVectorImpl(matId, 0, matCtx.getColNum, matCtx.getRowType)
    initMats()
  }

  /* pull functions */

  def pullWeightBias(indices: Array[Int], useAsync: Boolean): (IntFloatVector, IntFloatVector) = {
    if (useAsync) {
      val weightFuture = asyncPullWeight(indices)
      val biasFuture = asyncPullBias()
      (weightFuture.get.asInstanceOf[IntFloatVector], biasFuture.get.asInstanceOf[IntFloatVector])
    } else {
      (pullWeight(indices), pullBias())
    }
  }

  def pullWeightBiasEmbedding(indices: Array[Int], useAsync: Boolean): (IntFloatVector, IntFloatVector, Array[IntFloatVector]) = {
    if (useAsync) {
      val weightFuture = asyncPullWeight(indices)
      val biasFuture = asyncPullBias()
      val embeddingFuture = asyncPullEmbeddings(indices)

      (weightFuture.get.asInstanceOf[IntFloatVector], biasFuture.get.asInstanceOf[IntFloatVector],
        embeddingFuture.get().map(vectorFuture => vectorFuture.asInstanceOf[IntFloatVector]))
    } else {
      (pullWeight(indices), pullBias(), pullEmbeddings(indices))
    }
  }

  def pullWeightBiasEmbeddingMats(indices: Array[Int], useAsync: Boolean): (IntFloatVector, IntFloatVector, Array[IntFloatVector], IntFloatVector) = {
    if (useAsync) {
      val weightFuture = asyncPullWeight(indices)
      val biasFuture = asyncPullBias()
      val embeddingFuture = asyncPullEmbeddings(indices)
      val matsFuture = asyncPullMats()

      (weightFuture.get.asInstanceOf[IntFloatVector], biasFuture.get.asInstanceOf[IntFloatVector],
        embeddingFuture.get().map(vectorFuture => vectorFuture.asInstanceOf[IntFloatVector]),
        matsFuture.get().asInstanceOf[IntFloatVector])
    } else {
      (pullWeight(indices), pullBias(), pullEmbeddings(indices), pullMats())
    }
  }

  def pullBias(): IntFloatVector =
    bias.pull().asInstanceOf[IntFloatVector]

  def pullWeight(indices: Array[Int]): IntFloatVector =
    weights.pull(indices).asInstanceOf[IntFloatVector]

  def pullWeight(): IntFloatVector =
    weights.pull().asInstanceOf[IntFloatVector]

  def pullEmbeddings(indices: Array[Int]): Array[IntFloatVector] = {
    val rows = (0 until params.embeddingDim).toArray
    embedding.pull(rows, indices).map(f => f.asInstanceOf[IntFloatVector])
  }

  def pullEmbeddings(): Array[IntFloatVector] = {
    val rows = (0 until params.embeddingDim).toArray
    embedding.pull(rows).map(f => f.asInstanceOf[IntFloatVector])
  }

  def pullMats(): IntFloatVector =
    mats.pull().asInstanceOf[IntFloatVector]

  def asyncPullBias(): Future[Vector] =
    bias.asyncPull()

  def asyncPullWeight(indices: Array[Int]): Future[Vector] =
    weights.asyncPull(indices)

  def asyncPullEmbeddings(indices: Array[Int]): Future[Array[Vector]] = {
    val rows = (0 until params.embeddingDim).toArray
    embedding.asyncPull(rows, indices)
  }

  def asyncPullMats(): Future[Vector] =
    mats.asyncPull()

  /* push functions */
  def pushWeightBias(weightGrad: IntFloatVector, biasGrad: IntFloatVector, useAsync: Boolean): Unit = {
    if (useAsync) {
      asyncPushWeight(weightGrad)
      asyncPushBias(biasGrad)
    } else {
      pushWeight(weightGrad)
      pushBias(biasGrad)
    }
  }

  def pushWeightBiasEmbedding(weightGrad: IntFloatVector, biasGrad: IntFloatVector,
                              embeddingGrads: Array[IntFloatVector], useAsync: Boolean): Unit = {
    if (useAsync) {
      asyncPushWeight(weightGrad)
      asyncPushBias(biasGrad)
      asyncPushEmbedding(embeddingGrads)
    } else {
      pushWeight(weightGrad)
      pushBias(biasGrad)
      pushEmbedding(embeddingGrads)
    }
  }

  def pushWeightBiasEmbeddingMats(weightGrad: IntFloatVector, biasGrad: IntFloatVector,
                                  embeddingGrads: Array[IntFloatVector], matsGrad: IntFloatVector,
                                  useAsync: Boolean): Unit = {
    if (useAsync) {
      asyncPushWeight(weightGrad)
      asyncPushBias(biasGrad)
      asyncPushEmbedding(embeddingGrads)
      asyncPushMats(matsGrad)
    } else {
      pushWeight(weightGrad)
      pushBias(biasGrad)
      pushEmbedding(embeddingGrads)
      pushMats(matsGrad)
    }
  }

  def pushBias(grad: IntFloatVector): Unit =
    optim.update(bias, 1, grad)

  def pushWeight(grad: IntFloatVector): Unit =
    optim.update(weights, 1, grad)

  def pushEmbedding(grads: Array[IntFloatVector]): Unit = {
    val rows = (0 until params.embeddingDim).toArray
    optim.update(embedding, params.embeddingDim, rows, grads.map(f => f.asInstanceOf[Vector]))
  }

  def pushMats(grad: IntFloatVector): Unit =
    optim.update(mats, 1, grad)

  def asyncPushBias(grad: IntFloatVector): Future[VoidResult] =
    optim.asycUpdate(bias, 1, grad)

  def asyncPushWeight(grad: IntFloatVector): Future[VoidResult] =
    optim.asycUpdate(weights, 1, grad)

  def asyncPushEmbedding(grads: Array[IntFloatVector]): Future[VoidResult] = {
    val rows = (0 until params.embeddingDim).toArray
    optim.asycUpdate(embedding, params.embeddingDim, rows, grads.map(f => f.asInstanceOf[Vector]))
  }

  def asyncPushMats(grad: IntFloatVector): Future[VoidResult] =
    optim.asycUpdate(mats, 1, grad)

  /* making pytorch parameters/gradients functions */
  def makeBias(bias: IntFloatVector): Array[Float] = {
    val buf = new Array[Float](1)
    buf(0) = bias.get(0)
    buf
  }

  def makeBiasGrad(biasBuf: Array[Float], bias: IntFloatVector): Unit =
    bias.set(0, biasBuf(0))

  def makeWeights(weight: IntFloatVector, feats: Array[Long]): Array[Float] = {
    val buf = new Array[Float](feats.length)
    for (i <- feats.indices)
      buf(i) = weight.get(feats(i).toInt)
    buf
  }

  def makeWeights(weight: IntFloatVector): Array[Float] = {
    val buf = new Array[Float](weight.size())
    for (i <- 0 until weight.size())
      buf(i) = weight.get(i)
    buf
  }

  def makeWeightsGrad(weightsBuf: Array[Float], weights: IntFloatVector, feats: Array[Long]): Unit = {
    val grad = new Int2FloatOpenHashMap(weights.size())
    for (i <- weightsBuf.indices)
      grad.addTo(feats(i).toInt, weightsBuf(i))
    weights.setStorage(new IntFloatSparseVectorStorage(weights.dim().toInt, grad))
  }

  def makeEmbeddings(embeddings: Array[IntFloatVector], feats: Array[Long]): Array[Float] = {
    val buf = new Array[Float](feats.length * params.embeddingDim)
    for (i <- feats.indices)
      for (j <- 0 until params.embeddingDim)
        buf(i * params.embeddingDim + j) = embeddings(j).get(feats(i).toInt)
    buf
  }

  def makeEmbeddings(embeddings: Array[IntFloatVector]): Array[Float] = {
    val buf = new Array[Float](embeddings(0).size() * params.embeddingDim)
    for (i <- 0 until embeddings(0).size())
      for (j <- 0 until params.embeddingDim)
        buf(i * params.embeddingDim + j) = embeddings(j).get(i)
    buf
  }

  def makeEmbeddingGrad(embeddingBuf: Array[Float], embedding: Array[IntFloatVector], feats: Array[Long]): Unit = {
    val grads = embedding.map(f => f.size()).map(f => new Int2FloatOpenHashMap(f))
    for (i <- feats.indices) {
      for (j <- 0 until params.embeddingDim) {
        grads(j).addTo(feats(i).toInt, embeddingBuf(i * params.embeddingDim + j))
      }
    }

    embedding.zip(grads).foreach {
      case (e, g) =>
        e.setStorage(new IntFloatSparseVectorStorage(e.dim().toInt, g))
    }
  }

  def makeMats(mats: IntFloatVector): Array[Float] =
    mats.getStorage.asInstanceOf[IntFloatDenseVectorStorage].getValues

  def makeMatsGrad(matsBuf: Array[Float], mats: IntFloatVector): Unit =
    mats.setStorage(new IntFloatDenseVectorStorage(matsBuf))

  /* get pull indices functions */
  def distinctIntIndices(batch: CooLongFloatMatrix): Array[Int] = {
    val indices = new IntOpenHashSet()

    val cols = batch.getColIndices
    for (i <- 0 until cols.length)
      indices.add(cols(i).toInt)

    indices.toIntArray
  }

  /* optimize functions */
  def optimize(batch: Array[String]): Double = {
    val tuple3 = SampleParser.parse(batch, model.getType)
    val (coo, fields, targets) = (tuple3._1, tuple3._2, tuple3._3)

    val loss = model.getType match {
      case RecModelType.BIAS_WEIGHT =>
        optimizeBiasWeight(model, batch.length, coo, targets)
      case RecModelType.BIAS_WEIGHT_EMBEDDING =>
        optimizeBiasWeightEmbedding(model, batch.length, coo, targets)
      case RecModelType.BIAS_WEIGHT_EMBEDDING_MATS =>
        optimizeBiasWeightEmbeddingMats(model, batch.length, coo, targets)
      case RecModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        optimizeBiasWeightEmbeddingMatsField(model, batch.length, coo, fields, targets)
    }
    loss
  }

  def optimizeBiasWeight(model: RecModel, batchSize: Int, batch: CooLongFloatMatrix, targets: Array[Float]): Double = {
    val indices = distinctIntIndices(batch)

    incCallNum()
    var start = 0L

    // Pull parameters
    start = System.currentTimeMillis()
    val (weights, bias) = pullWeightBias(indices, useAsync)
    incPullTime(start)

    // Transfer the parameters formats from angel to pytorch
    start = System.currentTimeMillis()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    incMakeParamTime(start)

    // Calculate the gradients
    start = System.currentTimeMillis()
    val loss = model.backward(batchSize, batch, biasBuf, weightsBuf, targets)
    incCalTime(start)

    // Transfer the parameters formats from pytorch to angel
    start = System.currentTimeMillis()
    makeBiasGrad(biasBuf, bias)
    makeWeightsGrad(weightsBuf, weights, batch.getColIndices)
    incCalTime(start)

    // Push the gradient to ps
    start = System.currentTimeMillis()
    pushWeightBias(weights, bias, useAsync)
    incPushTime(start)

    loss * batchSize
  }

  def optimizeBiasWeightEmbedding(model: RecModel, batchSize: Int, batch: CooLongFloatMatrix, targets: Array[Float]): Double = {
    val indices = distinctIntIndices(batch)

    incCallNum()
    var start = 0L

    // Pull parameters
    start = System.currentTimeMillis()
    val (weights, bias, embedding) = pullWeightBiasEmbedding(indices, useAsync)
    incPullTime(start)

    // Transfer the parameters formats from angel to pytorch
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    incMakeParamTime(start)

    // Calculate the gradients
    start = System.currentTimeMillis()
    val loss = model.backward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf,
      params.embeddingDim, targets)
    incCalTime(start)

    // Transfer the parameters formats from pytorch to angel
    start = System.currentTimeMillis()
    makeBiasGrad(biasBuf, bias)
    makeWeightsGrad(weightsBuf, weights, batch.getColIndices)
    makeEmbeddingGrad(embeddingBuf, embedding, batch.getColIndices)
    incMakeGradTime(start)

    // Push the gradient to ps
    start = System.currentTimeMillis()
    pushWeightBiasEmbedding(weights, bias, embedding, useAsync)
    incPushTime(start)

    loss * batchSize
  }

  def optimizeBiasWeightEmbeddingMats(model: RecModel, batchSize: Int, batch: CooLongFloatMatrix, targets: Array[Float]): Double = {
    val indices = distinctIntIndices(batch)

    incCallNum()
    var start = 0L

    // Pull parameters
    start = System.currentTimeMillis()
    val (weights, bias, embedding, mats) = pullWeightBiasEmbeddingMats(indices, useAsync)
    incPullTime(start)

    // Transfer the parameters formats from angel to pytorch
    start = System.currentTimeMillis()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    val matsBuf = makeMats(mats)
    incMakeParamTime(start)

    // Calculate the gradients
    start = System.currentTimeMillis()
    val loss = model.backward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf,
      params.embeddingDim, matsBuf, params.matSizes, targets)
    incCalTime(start)

    // Transfer the parameters formats from pytorch to angel
    start = System.currentTimeMillis()
    makeBiasGrad(biasBuf, bias)
    makeWeightsGrad(weightsBuf, weights, batch.getColIndices)
    makeEmbeddingGrad(embeddingBuf, embedding, batch.getColIndices)
    makeMatsGrad(matsBuf, mats)
    incMakeGradTime(start)

    // Push the gradient to ps
    start = System.currentTimeMillis()
    pushWeightBiasEmbeddingMats(weights, bias, embedding, mats, useAsync)
    incPushTime(start)

    loss * batchSize
  }

  def optimizeBiasWeightEmbeddingMatsField(model: RecModel, batchSize: Int, batch: CooLongFloatMatrix, fields: Array[Long], targets: Array[Float]): Double = {
    var (start, end) = (0L, 0L)
    val indices = distinctIntIndices(batch)
    start = System.currentTimeMillis()
    val bias = pullBias()
    val weights = pullWeight(indices)
    val embedding = pullEmbeddings(indices)
    val mats = pullMats()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    val matsBuf = makeMats(mats)
    end = System.currentTimeMillis()
    val pullTime = end - start

    start = System.currentTimeMillis()
    val loss = model.backward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf,
      params.embeddingDim, matsBuf, params.matSizes, fields, targets)
    end = System.currentTimeMillis()
    val gradTime = end - start

    start = System.currentTimeMillis()
    makeBiasGrad(biasBuf, bias)
    makeWeightsGrad(weightsBuf, weights, batch.getColIndices)
    makeEmbeddingGrad(embeddingBuf, embedding, batch.getColIndices)
    makeMatsGrad(matsBuf, mats)
    pushBias(bias)
    pushWeight(weights)
    pushEmbedding(embedding)
    pushMats(mats)
    end = System.currentTimeMillis()

    val pushTime = end - start

    loss * batchSize
  }

  /* predict functions */

  def predict(batch: Array[String]): (Array[Float], Array[Float]) = {
    val tuple3 = SampleParser.parse(batch, model.getType)
    val (coo, fields, targets) = (tuple3._1, tuple3._2, tuple3._3)

    model.getType match {
      case RecModelType.BIAS_WEIGHT =>
        (targets, predictBiasWeight(model, batch.length, coo))
      case RecModelType.BIAS_WEIGHT_EMBEDDING =>
        (targets, predictBiasWeightEmbedding(model, batch.length, coo))
      case RecModelType.BIAS_WEIGHT_EMBEDDING_MATS =>
        (targets, predictBiasWeightEmbeddingMats(model, batch.length, coo))
      case RecModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
        (targets, predictBiasWeightEmbeddingMatsField(model, batch.length, coo, fields))
    }
  }

  def predictBiasWeight(model: RecModel, batchSize: Int, batch: CooLongFloatMatrix): Array[Float] = {
    val indices = distinctIntIndices(batch)
    val bias = pullBias()
    val weights = pullWeight(indices)
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    model.forward(batchSize, batch, biasBuf, weightsBuf)
  }

  def predictBiasWeightEmbedding(model: RecModel, batchSize: Int, batch: CooLongFloatMatrix): Array[Float] = {
    val indices = distinctIntIndices(batch)
    val bias = pullBias()
    val weights = pullWeight(indices)
    val embedding = pullEmbeddings(indices)
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    model.forward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf, params.embeddingDim)
  }

  def predictBiasWeightEmbeddingMats(model: RecModel, batchSize: Int, batch: CooLongFloatMatrix): Array[Float] = {
    val indices = distinctIntIndices(batch)
    val bias = pullBias()
    val weights = pullWeight(indices)
    val embedding = pullEmbeddings(indices)
    val mats = pullMats()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    val matsBuf = makeMats(mats)
    model.forward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf, params.embeddingDim,
      matsBuf, params.matSizes)
  }

  def predictBiasWeightEmbeddingMatsField(model: RecModel, batchSize: Int, batch: CooLongFloatMatrix, fields: Array[Long]): Array[Float] = {
    val indices = distinctIntIndices(batch)
    val bias = pullBias()
    val weights = pullWeight(indices)
    val embedding = pullEmbeddings(indices)
    val mats = pullMats()
    val biasBuf = makeBias(bias)
    val weightsBuf = makeWeights(weights, batch.getColIndices)
    val embeddingBuf = makeEmbeddings(embedding, batch.getColIndices)
    val matsBuf = makeMats(mats)
    model.forward(batchSize, batch, biasBuf, weightsBuf, embeddingBuf, params.embeddingDim,
      matsBuf, params.matSizes, fields)
  }

  /* time calculate functions */
  def incPullTime(startTs: Long): Unit = {
    totalPullTime = totalPullTime + (System.currentTimeMillis() - startTs)
  }

  def incPushTime(startTs: Long): Unit = {
    totalPushTime = totalPushTime + (System.currentTimeMillis() - startTs)
  }

  def incMakeParamTime(startTs: Long): Unit = {
    totalMakeParamTime = totalMakeParamTime + (System.currentTimeMillis() - startTs)
  }

  def incCalTime(startTs: Long): Unit = {
    totalCalTime = totalCalTime + (System.currentTimeMillis() - startTs)
  }

  def incMakeGradTime(startTs: Long): Unit = {
    totalMakeGradTime = totalMakeGradTime + (System.currentTimeMillis() - startTs)
  }

  def incCallNum(): Unit = {
    totalCallNum = totalCallNum + 1
  }

  def avgPullTime: Long = {
    totalPullTime / totalCallNum
  }

  def avgPushTime: Long = {
    totalPushTime / totalCallNum
  }

  def avgMakeParamTime: Long = {
    totalMakeParamTime / totalCallNum
  }

  def avgMakeGradTime: Long = {
    totalMakeGradTime / totalCallNum
  }

  def avgCalTime: Long = {
    totalCalTime / totalCallNum
  }
}
