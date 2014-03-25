package cc.factorie.app.nlp.embeddings

import cc.factorie.util.ClasspathURL
import cc.factorie.la
import java.util.zip.GZIPInputStream
import cc.factorie.app.nlp.embeddings

//object WordEmbedding extends embeddings.WordEmbedding(() => ClasspathURL.fromDirectory[WordEmbedding]("skip-gram-d100.W.gz").openConnection().getInputStream, 100,-1)

class WordEmbedding(val inputStreamFactory: () => java.io.InputStream, val dimensionSize: Int,numTake: Int = -1) extends scala.collection.mutable.LinkedHashMap[String,la.DenseTensor1] {
  def sourceFactory(): io.Source = io.Source.fromInputStream(new GZIPInputStream(inputStreamFactory()),"iso-8859-1")

  println("Reading Word Embeddings with dimension: %d".format(dimensionSize))

  initialize()
  def initialize() {
    val source = sourceFactory()
    var count = 0
    val lines = if(numTake > 0) source.getLines().take(numTake) else source.getLines()
    val firstLine = lines.next()
    val firstFields = firstLine.split("\\s+")
    val numLines = firstFields(0).toInt
    val dimension = firstFields(1).toInt
    assert(dimension == dimensionSize,"the specified dimension %d does not agree with the dimension %d given in the input file".format(dimension,dimensionSize))

    for (line <- lines) {
      val fields = line.split("\\s+")
      val tensor = new la.DenseTensor1(fields.drop(1).map(_.toDouble))
      assert(tensor.dim1 == dimensionSize,"the tensor has length " + tensor.dim1 + " , but it should have length + " + dimensionSize)
      this(fields(0)) = tensor
      count += 1
      if (count % 100000 == 0) println("word vector count: %d".format(count))
    }
    source.close()
  }

//  def close(string:String): Seq[String] = {
//    val t = this(string)
//    if (t eq null) return Nil
//    val top = new cc.factorie.util.TopN[String](10)
//    for ((s,t2) <- this) top.+=(0, t.dot(t2), s)
//    top.map(_.category)
//  }
}

trait WordEmbeddingOptions extends cc.factorie.util.CmdOptions  {
  val useEmbeddings = new CmdOption("use-embeddings",false,"BOOLEAN","Whether to use word embeddings")
  val embeddingFile = new CmdOption("embedding-file", "/home/vineet/canvas/embeddings/data/conll2003/", "STRING", "path to word2vec format file")
  val embeddingDim = new CmdOption("embedding-dim", 100, "INT", "embedding dimension")
  val embeddingScale = new CmdOption("embedding-scale", 10.0, "FLOAT", "The scale of the embeddings")
  val useOffsetEmbedding = new CmdOption("use-offset-embeddings", true, "BOOLEAN", "Whether to use offset embeddings")
  val numEmbeddingsToTake = new CmdOption("num-embeddings-to-take",-1,"INT","how many embeddings to take (assuming the file is sorted by word frequency. Default takes all of them")

}