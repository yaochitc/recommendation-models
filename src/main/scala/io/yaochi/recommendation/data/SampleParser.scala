package io.yaochi.recommendation.data

import java.lang.{Float => JFloat, Long => JLong}

import com.tencent.angel.exception.AngelException
import com.tencent.angel.ml.math2.matrix.CooLongFloatMatrix
import com.tencent.angel.ml.math2.vector.IntFloatVector
import com.tencent.angel.ml.math2.{MFactory, VFactory}
import io.yaochi.recommendation.model.RecModelType
import io.yaochi.recommendation.model.RecModelType.RecModelType
import it.unimi.dsi.fastutil.floats.FloatArrayList
import it.unimi.dsi.fastutil.longs.LongArrayList

object SampleParser {
  def parse(lines: Array[String], `type`: RecModelType): (CooLongFloatMatrix, Array[Long], Array[Float]) = `type` match {
    case RecModelType.BIAS_WEIGHT_EMBEDDING_MATS_FIELD =>
      parseLIBFFM(lines)
    case _ =>
      val tuple2 = parseLIBSVM(lines)
      (tuple2._1, null, tuple2._2)
  }

  def parseLIBSVM(lines: Array[String]): (CooLongFloatMatrix, Array[Float]) = {
    val rows = new LongArrayList()
    val cols = new LongArrayList()
    val values = new FloatArrayList()
    val targets = Array.ofDim[Float](lines.length)

    var index = 0
    for (i <- lines.indices) {
      val parts = lines(i).split(" ")
      val label = JFloat.parseFloat(parts(0))
      targets(i) = label

      for (j <- 1 until parts.length) {
        val kv = parts(j).split(":")
        val key = JLong.parseLong(kv(0)) - 1
        val value = JFloat.parseFloat(kv(1))

        rows.add(index)
        cols.add(key)
        values.add(value)
      }

      index += 1
    }

    val coo = MFactory.cooLongFloatMatrix(rows.toLongArray(),
      cols.toLongArray(), values.toFloatArray(), null)
    (coo, targets)
  }

  def parseLIBFFM(lines: Array[String]): (CooLongFloatMatrix, Array[Long], Array[Float]) = {
    val rows = new LongArrayList()
    val cols = new LongArrayList()
    val fields = new LongArrayList()
    val values = new FloatArrayList()
    val targets = Array.ofDim[Float](lines.length)

    var index = 0
    for (i <- lines.indices) {
      val parts = lines(i).split(" ")
      val label = JFloat.parseFloat(parts(0))
      targets(i) = label

      for (j <- 1 until parts.length) {
        val fkv = parts(j).split(":")
        val field = JLong.parseLong(fkv(0))
        val key = JLong.parseLong(fkv(1)) - 1
        val value = JFloat.parseFloat(fkv(2))

        rows.add(index)
        fields.add(field)
        cols.add(key)
        values.add(value)
      }

      index += 1
    }

    val coo = MFactory.cooLongFloatMatrix(rows.toLongArray(),
      cols.toLongArray(), values.toFloatArray(), null)

    (coo, fields.toLongArray(), targets)
  }

  def parseNodeFeature(line: String, dim: Int, format: String): (JLong, IntFloatVector) = {
    format match {
      case "sparse" =>
        parseSparseNodeFeature(line, dim)
      case "dense" =>
        parseDenseNodeFeature(line, dim)
      case _ =>
        throw new AngelException("format should be sparse or dense")
    }
  }

  def parseSparseNodeFeature(line: String, dim: Int): (JLong, IntFloatVector) = {
    if (line.length() == 0)
      return null

    val parts = line.split(" ")
    if (parts.length < 2)
      return null

    val node = JLong.parseLong(parts(0))
    val keys = Array.ofDim[Int](parts.length - 1)
    val values = Array.ofDim[Float](parts.length - 1)
    for (i <- 1 until parts.length) {
      val kv = parts(i).split(":")
      keys(i - 1) = Integer.parseInt(kv(0))
      values(i - 1) = JFloat.parseFloat(kv(1))
    }
    val feature = VFactory.sortedFloatVector(dim, keys, values)
    (node, feature)
  }

  def parseDenseNodeFeature(line: String, dim: Int): (JLong, IntFloatVector) = {
    if (line.length() == 0)
      return null

    val parts = line.split(" ")
    if (parts.length != dim + 1)
      throw new AngelException("number elements of data should be equal dim")

    val node = JLong.parseLong(parts(0))
    val values = (1 until parts.length).map(i => JFloat.parseFloat(parts(i))).toArray

    val feature = VFactory.denseFloatVector(values)
    (node, feature)
  }


  def parseFeature(line: String, dim: Int, format: String): IntFloatVector = {
    format match {
      case "sparse" =>
        parseSparseIntFloat(line, dim)
      case "dense" =>
        parseDenseIntFloat(line, dim)
      case _ =>
        throw new AngelException("format should be sparse or dense")
    }
  }

  def parseSparseIntFloat(line: String, dim: Int): IntFloatVector = {
    val parts = line.split(" ")

    val keys = Array.ofDim[Int](parts.length)
    val values = Array.ofDim[Float](parts.length)
    for (i <- parts.indices) {
      val kv = parts(i).split(":")
      keys(i) = Integer.parseInt(kv(0))
      if (keys(i) >= dim)
        throw new AngelException("feature index should be less than dim")
      values(i) = JFloat.parseFloat(kv(1))
    }

    VFactory.sortedFloatVector(dim, keys, values)
  }

  def parseDenseIntFloat(line: String, dim: Int): IntFloatVector = {
    val parts = line.split(" ")
    if (parts.length != dim)
      throw new AngelException("number elements of feature should be equal with dim")

    val vals = parts.map(JFloat.parseFloat)
    VFactory.denseFloatVector(vals)
  }

}
