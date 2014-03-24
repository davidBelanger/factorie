package cc.factorie.app.nlp.embeddings

import cc.factorie.util.ClasspathURL
import cc.factorie.la
import java.util.zip.GZIPInputStream
import cc.factorie.app.nlp.embeddings

object SkipGramEmbedding extends embeddings.SkipGramEmbedding(() => ClasspathURL.fromDirectory[SkipGramEmbedding]("skip-gram-d100.W.gz").openConnection().getInputStream, 100)

class SkipGramEmbedding(val inputStreamFactory: () => java.io.InputStream, dimensionSize: Int) extends scala.collection.mutable.LinkedHashMap[String,la.DenseTensor1] {
  def sourceFactory(): io.Source = io.Source.fromInputStream(new GZIPInputStream(inputStreamFactory()))

  println("Reading Word Embeddings with dimension: %d".format(dimensionSize))

  initialize()
  def initialize() {
    val source = sourceFactory()
    var count = 0
    val lines = source.getLines()
//    val firstLine = lines.next()
//    val firstFields = firstLine.split("\\s+")
//    val numLines = firstFields(0).toInt
//    val dimension = firstFields(1).toInt
//    assert(dimension == dimensionSize,"the specified dimension %d does not agree with the dimension %d given in the input file".format(dimension,dimensionSize))
    for (line <- lines) {
      println(line)
      val fields = line.split("\\s+")
      val tensor = new la.DenseTensor1(fields.drop(1).map(_.toDouble))
      assert(tensor.dim1 == dimensionSize,"the tensor has length " + tensor.dim1 + " , but it should have length + " + dimensionSize)
      this(fields(0)) = tensor
      count += 1
      if (count % 100000 == 0) println("word vector count: %d".format(count))
    }
    source.close()
  }

  def close(string:String): Seq[String] = {
    val t = this(string)
    if (t eq null) return Nil
    val top = new cc.factorie.util.TopN[String](10)
    for ((s,t2) <- this) top.+=(0, t.dot(t2), s)
    top.map(_.category)
  }
}