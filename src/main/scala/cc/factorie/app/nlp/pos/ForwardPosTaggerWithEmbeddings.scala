package cc.factorie.app.nlp.pos

import java.io._
import cc.factorie.app.classify.backend.{OptimizablePredictor, LinearMulticlassClassifier}
import cc.factorie.app.nlp.{load, Sentence, Token}
import cc.factorie.{la, optimize}
import cc.factorie.util._
import cc.factorie.la._
import cc.factorie.optimize.Trainer
import cc.factorie.variable.{TensorVar, LabeledMutableDiscreteVar}
import cc.factorie.model.{Parameters, DotTemplateWithStatistics2}
import cc.factorie.app.nlp.embeddings.{WordEmbeddingOptions, WordEmbedding}


class SparseAndDenseLinearMulticlassClassifier(labelSize: Int, sparseFeatureSize: Int, denseFeatureSize: Int) extends cc.factorie.app.classify.backend.MulticlassClassifier[(SparseBinaryTensor1,DenseTensor1)] with Parameters with OptimizablePredictor[Tensor1,(SparseBinaryTensor1,DenseTensor1)] {
  val weightsForSparseFeatures = Weights(new DenseTensor2(sparseFeatureSize, labelSize))
  val weightsForDenseFeatures = Weights(new DenseTensor2(denseFeatureSize, labelSize))


  def predict(features: (SparseBinaryTensor1,DenseTensor1)): Tensor1 = {val result = weightsForSparseFeatures.value.leftMultiply(features._1); result.+=(weightsForDenseFeatures.value.leftMultiply(features._2)); result}
  def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, features: (SparseBinaryTensor1,DenseTensor1), gradient: Tensor1, weight: Double) = {
    accumulator.accumulate(weightsForSparseFeatures, features._1 outer gradient)
    accumulator.accumulate(weightsForDenseFeatures, features._2  outer gradient)
  }

}


class SparseAndDenseLinearMulticlassClassifierExample(model: SparseAndDenseLinearMulticlassClassifier,targetIntValue: Int,sparseFeatures: SparseBinaryTensor1, denseFeatures: DenseTensor1,lossAndGradient: optimize.OptimizableObjectives.Multiclass, weight: Double = 1.0) extends optimize.Example {
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator) {
    val input = (sparseFeatures,denseFeatures)
    val prediction = model.predict(input)
    val (obj, ograd) = lossAndGradient.valueAndGradient(prediction, targetIntValue)
    if (value != null) value.accumulate(obj * weight)
    if (gradient != null) model.accumulateObjectiveGradient(gradient, input, ograd, weight)
  }
}

class ForwardPosTaggerWithEmbeddings(embedding: WordEmbedding) extends GeneralForwardPosTagger2{
  // Different ways to load saved parameters
//  def this(stream:InputStream) = { this(); deserialize(stream) }
//  def this(file: File) = this(new FileInputStream(file))
//  def this(url:java.net.URL) = {
//    this()
//    val stream = url.openConnection.getInputStream
//    if (stream.available <= 0) throw new Error("Could not open "+url)
//    println("ForwardPosTagger loading from "+url)
//    deserialize(stream)
//  }
  val denseFeatureDomainSize = if (embedding == null) 1 else embedding.dimensionSize
  val defaultEmbedding = new DenseTensor1(denseFeatureDomainSize)
  def getEmbedding(str: String) : DenseTensor1 = {
    if (embedding == null)
      defaultEmbedding
    else {
      if(embedding.contains(str))
        embedding(str)
      else
        defaultEmbedding
    }

  }
  lazy val model = new SparseAndDenseLinearMulticlassClassifier(PennPosDomain.size, FeatureDomain.dimensionSize,denseFeatureDomainSize)

  def getFeatures(token: Token, index: Int, lemmaStrings: Lemmas): (SparseBinaryTensor1,DenseTensor1) = {
    val sparseFeatures = features(token, index, lemmaStrings)
    val denseFeatures = getEmbedding(lemmaStrings(index)) //todo: is this the right index?
    (sparseFeatures,denseFeatures)
  }

  def predictToken(token: Token, index: Int, lemmaStrings: Lemmas): Int = {
    model.classification(getFeatures(token,index,lemmaStrings)).bestLabelIndex
  }


  class SentenceClassifierExample(val tokens:Seq[Token], model:SparseAndDenseLinearMulticlassClassifier, lossAndGradient: optimize.OptimizableObjectives.Multiclass) extends optimize.Example {
    def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator) {
      val lemmaStrings = lemmas(tokens)
      for (index <- 0 until tokens.length) {
        val token = tokens(index)
        val posLabel = token.attr[LabeledPennPosTag]
        val featureVector = getFeatures(token, index, lemmaStrings)
        new optimize.PredictorExample(model, featureVector, posLabel.target.intValue, lossAndGradient, 1.0).accumulateValueAndGradient(value, gradient)
        if (exampleSetsToPrediction) {
          posLabel.set(model.classification(featureVector).bestLabelIndex)(null)
        }
      }
    }
  }

  def train(trainSentences:Seq[Sentence], testSentences:Seq[Sentence], lrate:Double = 0.1, decay:Double = 0.01, cutoff:Int = 2, doBootstrap:Boolean = true, useHingeLoss:Boolean = false, numIterations: Int = 5, l1Factor:Double = 0.000001, l2Factor:Double = 0.000001)(implicit random: scala.util.Random) {
    // TODO Accomplish this TokenNormalization instead by calling POS3.preProcess
    //for (sentence <- trainSentences ++ testSentences; token <- sentence.tokens) cc.factorie.app.nlp.segment.PlainTokenNormalizer.processToken(token)

    val toksPerDoc = 5000
    WordData.computeWordFormsByDocumentFrequency(trainSentences.flatMap(_.tokens), 1, toksPerDoc)
    WordData.computeAmbiguityClasses(trainSentences.flatMap(_.tokens))

    // Prune features by count
    FeatureDomain.dimensionDomain.gatherCounts = true
    for (sentence <- trainSentences) features(sentence.tokens) // just to create and count all features
    FeatureDomain.dimensionDomain.trimBelowCount(cutoff)
    FeatureDomain.freeze()
    println("After pruning using %d features.".format(FeatureDomain.dimensionDomain.size))

    /* Print out some features (for debugging) */
    //println("ForwardPosTagger.train\n"+trainSentences(3).tokens.map(_.string).zip(features(trainSentences(3).tokens).map(t => new FeatureVariable(t).toString)).mkString("\n"))


    def evaluate() {
      exampleSetsToPrediction = doBootstrap
      printAccuracy(trainSentences, "Training: ")
      printAccuracy(testSentences, "Testing: ")
      val sparsity = model.weightsForDenseFeatures.value.toSeq.count(_ == 0).toFloat/ model.weightsForDenseFeatures.value.length
      println("Sparsity: " + sparsity)
    }
    val examples = trainSentences.shuffle.par.map(sentence =>
      new SentenceClassifierExample(sentence.tokens, model, if (useHingeLoss) cc.factorie.optimize.OptimizableObjectives.hingeMulticlass else cc.factorie.optimize.OptimizableObjectives.sparseLogMulticlass)).seq
    //val optimizer = new cc.factorie.optimize.AdaGrad(rate=lrate)
    val optimizer = new cc.factorie.optimize.AdaGradRDA(rate=lrate, l1=l1Factor/examples.length, l2=l2Factor/examples.length)
    Trainer.onlineTrain(model.parameters, examples, maxIterations=numIterations, optimizer=optimizer, evaluate=evaluate, useParallelTrainer = false)
    if (false) {
      // Print test results to file
      val source = new java.io.PrintStream(new File("pos1-test-output.txt"))
      for (s <- testSentences) {
        for (t <- s.tokens) { val p = t.attr[LabeledPennPosTag]; source.println("%s %20s  %6s %6s".format(if (p.valueIsTarget) " " else "*", t.string, p.target.categoryValue, p.categoryValue)) }
        source.println()
      }
      source.close()
    }
  }


  def serialize(stream: java.io.OutputStream): Unit = ???
  //{
  //  import CubbieConversions._
//    val sparseEvidenceWeights = new la.DenseLayeredTensor2(model.weights.value.dim1, model.weights.value.dim2, new la.SparseIndexedTensor1(_))
//    model.weights.value.foreachElement((i, v) => if (v != 0.0) sparseEvidenceWeights += (i, v))
//    model.weights.set(sparseEvidenceWeights)
//    val dstream = new java.io.DataOutputStream(new BufferedOutputStream(stream))
//    BinarySerializer.serialize(FeatureDomain.dimensionDomain, dstream)
//    BinarySerializer.serialize(model, dstream)
//    BinarySerializer.serialize(WordData.ambiguityClasses, dstream)
//    BinarySerializer.serialize(WordData.sureTokens, dstream)
//    BinarySerializer.serialize(WordData.docWordCounts, dstream)
//    dstream.close()  // TODO Are we really supposed to close here, or is that the responsibility of the caller
  //}
  def deserialize(stream: java.io.InputStream): Unit = ???
  //{
  //  import CubbieConversions._
//    val dstream = new java.io.DataInputStream(new BufferedInputStream(stream))
//    BinarySerializer.deserialize(FeatureDomain.dimensionDomain, dstream)
//    model.weights.set(new la.DenseLayeredTensor2(FeatureDomain.dimensionDomain.size, PennPosDomain.size, new la.SparseIndexedTensor1(_)))
//    BinarySerializer.deserialize(model, dstream)
//    BinarySerializer.deserialize(WordData.ambiguityClasses, dstream)
//    BinarySerializer.deserialize(WordData.sureTokens, dstream)
//    BinarySerializer.deserialize(WordData.docWordCounts, dstream)
//    dstream.close()  // TODO Are we really supposed to close here, or is that the responsibility of the caller
//  }

}

class ForwardPosWithEmbeddingsOptions extends ForwardPosOptions with WordEmbeddingOptions

object ForwardPosTrainerWithEmbeddingsTrainer extends HyperparameterMain {
  def evaluateParameters(args: Array[String]): Double = {
    implicit val random = new scala.util.Random(0)
    val opts = new ForwardPosWithEmbeddingsOptions
    opts.parse(args)
    assert(opts.trainFile.wasInvoked || opts.trainDir.wasInvoked || opts.trainFiles.wasInvoked)
    // Expects three command-line arguments: a train file, a test file, and a place to save the model
    // the train and test files are supposed to be in OWPL format
    val useEmbeddings = opts.useEmbeddings.value
    if(useEmbeddings) println("using embeddings") else println("not using embeddings")
    val embedding = if(useEmbeddings)  new WordEmbedding(() => new FileInputStream(opts.embeddingFile.value),opts.embeddingDim.value,opts.numEmbeddingsToTake.value) else null
    val pos = new  ForwardPosTaggerWithEmbeddings(embedding)

    assert(!(opts.trainDir.wasInvoked && opts.trainFiles.wasInvoked))
    var trainFileList = Seq(opts.trainFile.value)
    if(opts.trainDir.wasInvoked){
      trainFileList = FileUtils.getFileListFromDir(opts.trainDir.value)
    } else if (opts.trainFiles.wasInvoked){
      trainFileList =  opts.trainFiles.value.split(",")
    }

    assert(!(opts.testDir.wasInvoked && opts.testFiles.wasInvoked))
    var testFileList = Seq(opts.testFile.value)
    if(opts.testDir.wasInvoked){
      testFileList = FileUtils.getFileListFromDir(opts.testDir.value)
    }else if (opts.testFiles.wasInvoked){
      testFileList =  opts.testFiles.value.split(",")
    }

    val trainDocs = trainFileList.map(fname => {
      if(opts.wsj.value) load.LoadWSJMalt.fromFilename(fname).head
      else load.LoadOntonotes5.fromFilename(fname).head
    })
    val testDocs = testFileList.map(fname => {
      if(opts.wsj.value) load.LoadWSJMalt.fromFilename(fname).head
      else load.LoadOntonotes5.fromFilename(fname).head
    })

    //for (d <- trainDocs) println("POS3.train 1 trainDoc.length="+d.length)
    println("Read %d training tokens from %d files.".format(trainDocs.map(_.tokenCount).sum, trainDocs.size))
    println("Read %d testing tokens from %d files.".format(testDocs.map(_.tokenCount).sum, testDocs.size))

    val trainPortionToTake = if(opts.trainPortion.wasInvoked) opts.trainPortion.value else 1.0
    val testPortionToTake =  if(opts.testPortion.wasInvoked) opts.testPortion.value else 1.0
    val trainSentencesFull = trainDocs.flatMap(_.sentences)
    val trainSentences = trainSentencesFull.take((trainPortionToTake*trainSentencesFull.length).floor.toInt)
    val testSentencesFull = testDocs.flatMap(_.sentences)
    val testSentences = testSentencesFull.take((testPortionToTake*testSentencesFull.length).floor.toInt)

    pos.train(trainSentences, testSentences,
      opts.rate.value, opts.delta.value, opts.cutoff.value, opts.updateExamples.value, opts.useHingeLoss.value, numIterations=opts.numIters.value.toInt,l1Factor=opts.l1.value, l2Factor=opts.l2.value)
    if (opts.saveModel.value) {
      pos.serialize(opts.modelFile.value)
      val pos2 = new ForwardPosTagger
      pos2.deserialize(new java.io.File(opts.modelFile.value))
      pos.printAccuracy(testDocs.flatMap(_.sentences), "pre-serialize accuracy: ")
      pos2.printAccuracy(testDocs.flatMap(_.sentences), "post-serialize accuracy: ")
    }
    val acc = pos.accuracy(testDocs.flatMap(_.sentences))._1
    if(opts.targetAccuracy.wasInvoked) cc.factorie.assertMinimalAccuracy(acc,opts.targetAccuracy.value.toDouble)
    acc
  }
}


object ForwardPosWithEmbeddingsOptimizer {
  def main(args: Array[String]) {
    val opts = new ForwardPosWithEmbeddingsOptions
    opts.parse(args)
    opts.saveModel.setValue(false)
    val l1 = cc.factorie.util.HyperParameter(opts.l1, new cc.factorie.util.LogUniformDoubleSampler(1e-10, 1e2))
    val l2 = cc.factorie.util.HyperParameter(opts.l2, new cc.factorie.util.LogUniformDoubleSampler(1e-10, 1e2))
    val rate = cc.factorie.util.HyperParameter(opts.rate, new cc.factorie.util.LogUniformDoubleSampler(1e-4, 1e4))
    val delta = cc.factorie.util.HyperParameter(opts.delta, new cc.factorie.util.LogUniformDoubleSampler(1e-4, 1e4))
    val cutoff = cc.factorie.util.HyperParameter(opts.cutoff, new cc.factorie.util.SampleFromSeq(List(0,1,2,3)))
    /*
    val ssh = new cc.factorie.util.SSHActorExecutor("apassos",
      Seq("avon1", "avon2"),
      "/home/apassos/canvas/factorie-test",
      "try-log/",
      "cc.factorie.app.nlp.parse.DepParser2",
      10, 5)
      */
    val qs = new cc.factorie.util.QSubExecutor(30, "cc.factorie.app.nlp.pos.ForwardPosTrainerWithEmbeddingsTrainer")
    val optimizer = new cc.factorie.util.HyperParameterSearcher(opts, Seq(l1, l2, rate, delta, cutoff), qs.execute, 200, 180, 60)
    val result = optimizer.optimize()
    println("Got results: " + result.mkString(" "))
    println("Best l1: " + opts.l1.value + " best l2: " + opts.l2.value)
    opts.saveModel.setValue(true)
    println("Running best configuration...")
    import scala.concurrent.duration._
    import scala.concurrent.Await
    Await.result(qs.execute(opts.values.flatMap(_.unParse).toArray), 5.hours)
    println("Done")
  }
}