package cc.factorie.app.nlp.parse

import cc.factorie.app.nlp._
import cc.factorie._
import embeddings.{WordEmbeddingOptions, WordEmbedding}
import model.WeightsSet
import pos.{LabeledPennPosTag, SparseAndDenseLinearMulticlassClassifier, PennPosTag}
import cc.factorie.app.nlp.parse
import parse.WSJTransitionBasedParser
import scala.collection.mutable.{HashMap, ArrayBuffer}
import scala.util.parsing.json.JSON
import scala.annotation.tailrec
import java.io._
import cc.factorie.util._
import scala._
import cc.factorie.optimize._
import scala.concurrent.Await
import cc.factorie.variable.LabeledCategoricalVariable
import cc.factorie.variable.BinaryFeatureVectorVariable
import cc.factorie.variable.CategoricalVectorDomain
import cc.factorie.variable.LabeledCategoricalVariable
import cc.factorie.variable.BinaryFeatureVectorVariable
import cc.factorie.variable.CategoricalVectorDomain
import cc.factorie.variable.{LabeledCategoricalVariable, BinaryFeatureVectorVariable, CategoricalVectorDomain, CategoricalDomain}
import scala.collection.mutable
import cc.factorie.app.classify.backend._
import scala.Some
import scala.Some
import cc.factorie.app.nlp.DocumentAnnotator
import cc.factorie.variable.CategoricalDomain
import cc.factorie.app.classify.backend.LinearMulticlassClassifier
import cc.factorie.app.classify.backend.MulticlassClassifierTrainer
import cc.factorie.app.nlp.Sentence
import cc.factorie.optimize.AdaGradRDA
import cc.factorie.app.classify.backend.OnlineLinearMulticlassTrainer
import cc.factorie.app.nlp.Document
import cc.factorie.app.nlp.lemma
import cc.factorie.app.nlp.Token
import cc.factorie.app.nlp.SharedNLPCmdOptions
import cc.factorie.app.nlp.load
import cc.factorie.app.classify.backend.SVMMulticlassTrainer
import cc.factorie.optimize.OptimizableObjectives
import cc.factorie.app.nlp.DocumentAnnotator
import scala.Some
import scala.Some
import cc.factorie.la
import cc.factorie.app.classify.backend.LinearMulticlassClassifier
import cc.factorie.app.classify.backend.MulticlassClassifierTrainer
import cc.factorie.app.nlp.Sentence
import cc.factorie.optimize.AdaGradRDA
import cc.factorie.optimize
import cc.factorie.app.classify.backend.OnlineLinearMulticlassTrainer
import cc.factorie.app.nlp.DocumentAnnotator
import cc.factorie.variable.CategoricalDomain
import cc.factorie.app.nlp.Document
import cc.factorie.app.nlp.lemma
import cc.factorie.app.nlp.Token
import cc.factorie.app.nlp.load
import cc.factorie.app.classify.backend.SVMMulticlassTrainer
import cc.factorie.optimize.OptimizableObjectives
import cc.factorie.la.{WeightsMapAccumulator, SparseBinaryTensor1, DenseTensor1}
import cc.factorie.DenseTensor1


class TransitionBasedParserWithEmbeddings(embedding: WordEmbedding) extends BaseTransitionBasedParser {
//  def this(stream:InputStream) = { this(); deserialize(stream) }
//  def this(file: File) = this(new FileInputStream(file))
//  def this(url:java.net.URL) = {
//    this()
//    val stream = url.openConnection.getInputStream
//    if (stream.available <= 0) throw new Error("Could not open "+url)
//    println("TransitionBasedParser loading from "+url)
//    deserialize(stream)
//  }
  def serialize(stream: java.io.OutputStream): Unit = {

  }
  def deserialize(stream: java.io.InputStream): Unit = {

  }

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
  lazy val model = new SparseAndDenseLinearMulticlassClassifier(labelDomain.size, featuresDomain.dimensionSize,denseFeatureDomainSize)

  def getFeatures(v: ParseDecisionVariable): (SparseBinaryTensor1,DenseTensor1) =  {
    val denseFeatures =  getDenseFeatures(v)
    (v.features.value.asInstanceOf[SparseBinaryTensor1],denseFeatures)
  }
  def classify(v: ParseDecisionVariable) = getParseDecision(labelDomain.category(model.classification(getFeatures(v)).bestLabelIndex))
  def getDenseFeatures(v: ParseDecisionVariable): DenseTensor1 = getEmbedding(v.state.stackToken(0).form)

  def trainFromVariables(vs: Iterable[ParseDecisionVariable], trainer: Trainer, evaluate: (SparseAndDenseLinearMulticlassClassifier) => Unit) {
//class SparseAndDenseLinearMulticlassClassifierExample(model: SparseAndDenseLinearMulticlassClassifier,targetIntValue: Int,sparseFeatures: SparseBinaryTensor1, denseFeatures: DenseTensor1,lossAndGradient: optimize.OptimizableObjectives.Multiclass, weight: Double = 1.0) extends optimize.Example {

    trainer.trainFromExamples(vs.map(v => {
      val denseFeatures =getDenseFeatures(v)
      new pos.SparseAndDenseLinearMulticlassClassifierExample(model,v.target.intValue,v.features.value.asInstanceOf[SparseBinaryTensor1],denseFeatures,optimize.OptimizableObjectives.sparseLogMulticlass,1.0)
    }))
  }

//  def train(trainSentences:Iterable[Sentence], testSentences:Iterable[Sentence], numBootstrappingIterations:Int = 2, l1Factor:Double = 0.00001, l2Factor:Double = 0.00001, nThreads: Int = 1)(implicit random: scala.util.Random): Unit = {
//    featuresDomain.dimensionDomain.gatherCounts = true
//    var trainingVars: Iterable[ParseDecisionVariable] = generateDecisions(trainSentences, ParserConstants.TRAINING, nThreads)
//    println("Before pruning # features " + featuresDomain.dimensionDomain.size)
//    println("TransitionBasedParser.train first 20 feature counts: "+featuresDomain.dimensionDomain.counts.toSeq.take(20))
//    featuresDomain.dimensionDomain.trimBelowCount(5) // Every feature is actually counted twice, so this removes features that were seen 2 times or less
//    featuresDomain.freeze()
//    println("After pruning # features " + featuresDomain.dimensionSize)
//    trainingVars = generateDecisions(trainSentences, ParserConstants.TRAINING, nThreads)
//
//    val numTrainSentences = trainSentences.size
//    val optimizer = new AdaGradRDA(1.0, 0.1, l1Factor / numTrainSentences, l2Factor / numTrainSentences)
//
//    trainDecisions(trainingVars, optimizer, trainSentences, testSentences)
//    trainingVars = null // Allow them to be GC'ed
//    for (i <- 0 until numBootstrappingIterations) {
//      println("Boosting iteration " + (i+1))
//      trainDecisions(generateDecisions(trainSentences, ParserConstants.BOOSTING, nThreads), optimizer, trainSentences, testSentences)
//    }
//  }
//
//  def trainDecisions(trainDecisions:Iterable[ParseDecisionVariable], optimizer:optimize.GradientOptimizer, trainSentences:Iterable[Sentence], testSentences:Iterable[Sentence])(implicit random: scala.util.Random): Unit = {
//    def evaluate(c: SparseAndDenseLinearMulticlassClassifier) {
//      println(model.weightsForSparseFeatures.value.toSeq.count(_ == 0).toFloat/model.weightsForSparseFeatures.value.length +" sparsity")
//      println(" TRAIN "+testString(trainSentences))
//      println(" TEST  "+testString(testSentences))
//    }
//    new OnlineLinearMulticlassTrainer(optimizer=optimizer, maxIterations=2).baseTrain(model, trainDecisions.map(_.target.intValue).toSeq, trainDecisions.map(_.features.value).toSeq, trainDecisions.map(v => 1.0).toSeq, evaluate=evaluate)
//  }

  def boosting(ss: Iterable[Sentence], nThreads: Int, trainer: Trainer, evaluate: SparseAndDenseLinearMulticlassClassifier => Unit) =
    trainFromVariables(generateDecisions(ss, ParserConstants.BOOSTING, nThreads), trainer, evaluate)
}


class TransitionBasedParserWithEmbeddingsArgs extends TransitionBasedParserArgs with WordEmbeddingOptions

object TransitionBasedParserWithEmbeddingsTrainer extends cc.factorie.util.HyperparameterMain {
  def evaluateParameters(args: Array[String]) = {
    val opts = new TransitionBasedParserWithEmbeddingsArgs
    implicit val random = new scala.util.Random(0)
    opts.parse(args)

    assert(opts.trainFiles.wasInvoked || opts.trainDir.wasInvoked)

    // Load the sentences
    def loadSentences(listOpt: opts.CmdOption[List[String]], dirOpt: opts.CmdOption[String]): Seq[Sentence] = {
      var fileList = Seq.empty[String]
      if (listOpt.wasInvoked) fileList = listOpt.value.toSeq
      if (dirOpt.wasInvoked) fileList ++= FileUtils.getFileListFromDir(dirOpt.value)
      fileList.flatMap(fname => {
        if(opts.wsj.value)
          load.LoadWSJMalt.fromFilename(fname, loadLemma=load.AnnotationTypes.AUTO, loadPos=load.AnnotationTypes.AUTO).head.sentences.toSeq
        else if (opts.ontonotes.value)
          load.LoadOntonotes5.fromFilename(fname, loadLemma=load.AnnotationTypes.AUTO, loadPos=load.AnnotationTypes.AUTO).head.sentences.toSeq
        else
          load.LoadConll2008.fromFilename(fname).head.sentences.toSeq
      })
    }

    val sentencesFull = loadSentences(opts.trainFiles, opts.trainDir)
    val devSentencesFull = loadSentences(opts.devFiles, opts.devDir)
    val testSentencesFull = loadSentences(opts.testFiles, opts.testDir)

    val trainPortionToTake = if(opts.trainPortion.wasInvoked) opts.trainPortion.value.toDouble  else 1.0
    val testPortionToTake =  if(opts.testPortion.wasInvoked) opts.testPortion.value.toDouble  else 1.0
    val sentences = sentencesFull.take((trainPortionToTake*sentencesFull.length).floor.toInt)
    val testSentences = testSentencesFull.take((testPortionToTake*testSentencesFull.length).floor.toInt)
    val devSentences = devSentencesFull.take((testPortionToTake*devSentencesFull.length).floor.toInt)

    println("Total train sentences: " + sentences.size)
    println("Total test sentences: " + testSentences.size)

    def testSingle(c: BaseTransitionBasedParser, ss: Seq[Sentence], extraText: String = ""): Unit = {
      if (ss.nonEmpty) {
        println(extraText + " " + c.testString(ss))
      }
    }

    def testAll(c: BaseTransitionBasedParser, extraText: String = ""): Unit = {
      println("\n")
      testSingle(c, sentences,     "Train " + extraText)
      testSingle(c, devSentences,  "Dev "   + extraText)
      testSingle(c, testSentences, "Test "  + extraText)
    }

    // Load other parameters
    val numBootstrappingIterations = opts.bootstrapping.value.toInt
    val useEmbeddings = opts.useEmbeddings.value
    if(useEmbeddings) println("using embeddings") else println("not using embeddings")
    val embedding = if(useEmbeddings)  new WordEmbedding(() => new FileInputStream(opts.embeddingFile.value),opts.embeddingDim.value,opts.numEmbeddingsToTake.value) else null

    val c = new TransitionBasedParserWithEmbeddings(embedding)
    val l1 = 2*opts.l1.value / sentences.length
    val l2 = 2*opts.l2.value / sentences.length
    val optimizer = new AdaGradRDA(opts.rate.value, opts.delta.value, l1, l2)
    println(s"Initializing trainer (${opts.nThreads.value} threads)")
//    val trainer2 = if (opts.useSVM.value) new SVMMulticlassTrainer(opts.nThreads.value)
//    else new OnlineLinearMulticlassTrainer(optimizer=optimizer, useParallel=if (opts.nThreads.value > 1) true else false, nThreads=opts.nThreads.value, objective=OptimizableObjectives.hingeMulticlass, maxIterations=opts.maxIters.value)

    val trainer = new OnlineTrainer(c.model.parameters,optimizer)

    def evaluate(cls: SparseAndDenseLinearMulticlassClassifier) {
      println(cls.weightsForSparseFeatures.value.toSeq.count(x => x == 0).toFloat/cls.weightsForSparseFeatures.value.length +" sparsity")
      testAll(c, "iteration ")
    }
    c.featuresDomain.dimensionDomain.gatherCounts = true
    println("Generating decisions...")
    c.generateDecisions(sentences, c.ParserConstants.TRAINING, opts.nThreads.value)

    println("Before pruning # features " + c.featuresDomain.dimensionDomain.size)
    c.featuresDomain.dimensionDomain.trimBelowCount(2*opts.cutoff.value)
    c.featuresDomain.freeze()
    c.featuresDomain.dimensionDomain.gatherCounts = false
    println("After pruning # features " + c.featuresDomain.dimensionDomain.size)
    println("Training...")

    var trainingVs = c.generateDecisions(sentences, c.ParserConstants.TRAINING, opts.nThreads.value)

    /* Print out features */
    //    sentences.take(5).foreach(s => {
    //      println(s"Sentence: ${s.tokens.map(_.string).mkString(" ")}")
    //      val trainingVariables = c.generateDecisions(Seq(s), c.ParserConstants.TRAINING, opts.nThreads.value)
    //      trainingVariables.foreach(tv => {
    //        println(s"Training decision: ${
    //          val transition = tv.categoryValue.split(" ")
    //          transition.take(2).map(x => c.ParserConstants.getString(x.toInt)).mkString(" ") + " " + transition(2)
    //        }; features: ${
    //          tv.features.domain.dimensionDomain.categories.zip(tv.features.value.toSeq).filter(_._2 == 1.0).map(_._1).mkString(" ")
    //        }")
    //      })
    //      println()
    //    })

    c.trainFromVariables(trainingVs, trainer, evaluate)

    trainingVs = null // GC the old training labels
    for (i <- 0 until numBootstrappingIterations) {
      println("Boosting iteration " + i)
      c.boosting(sentences, nThreads=opts.nThreads.value, trainer=trainer, evaluate=evaluate)
    }

    testSentences.par.foreach(c.process)

    if (opts.saveModel.value) {
      val modelUrl: String = if (opts.modelDir.wasInvoked) opts.modelDir.value else opts.modelDir.defaultValue + System.currentTimeMillis().toString + ".factorie"
      c.serialize(new java.io.File(modelUrl))
      val d = new TransitionBasedParserWithEmbeddings(embedding)
      d.deserialize(new java.io.File(modelUrl))
      testSingle(d, testSentences, "Post serialization accuracy ")
    }
    val testLAS = ParserEval.calcLas(testSentences.map(_.attr[ParseTree]))
    if(opts.targetAccuracy.wasInvoked) cc.factorie.assertMinimalAccuracy(testLAS,opts.targetAccuracy.value.toDouble)

    testLAS
  }
}



object TransitionBasedParserWithEmbeddingsOptimizer {
  def main(args: Array[String]) {
    val opts = new TransitionBasedParserArgs
    opts.parse(args)
    opts.saveModel.setValue(false)
    val l1 = cc.factorie.util.HyperParameter(opts.l1, new cc.factorie.util.LogUniformDoubleSampler(1e-10, 1e2))
    val l2 = cc.factorie.util.HyperParameter(opts.l2, new cc.factorie.util.LogUniformDoubleSampler(1e-10, 1e2))
    val rate = cc.factorie.util.HyperParameter(opts.rate, new cc.factorie.util.LogUniformDoubleSampler(1e-4, 1e4))
    val delta = cc.factorie.util.HyperParameter(opts.delta, new cc.factorie.util.LogUniformDoubleSampler(1e-4, 1e4))
    val cutoff = cc.factorie.util.HyperParameter(opts.cutoff, new cc.factorie.util.SampleFromSeq[Int](Seq(0, 1, 2)))
    val bootstrap = cc.factorie.util.HyperParameter(opts.bootstrapping, new cc.factorie.util.SampleFromSeq[Int](Seq(0, 1, 2)))
    val maxit = cc.factorie.util.HyperParameter(opts.maxIters, new cc.factorie.util.SampleFromSeq[Int](Seq(2, 5, 8)))
    /*
    val ssh = new cc.factorie.util.SSHActorExecutor("apassos",
      Seq("avon1", "avon2"),
      "/home/apassos/canvas/factorie-test",
      "try-log/",
      "cc.factorie.app.nlp.parse.TransitionBasedParser",
      10, 5)
      */
    val qs = new cc.factorie.util.QSubExecutor(32, "cc.factorie.app.nlp.parse.TransitionBasedParserWithEmbeddingsTrainer")
    val optimizer = new cc.factorie.util.HyperParameterSearcher(opts, Seq(l1, l2, rate, delta, cutoff, bootstrap, maxit), qs.execute, 200, 180, 60)
    val result = optimizer.optimize()
    println("Got results: " + result.mkString(" "))
    opts.saveModel.setValue(true)
    println("Running best configuration...")
    import scala.concurrent.duration._
    Await.result(qs.execute(opts.values.flatMap(_.unParse).toArray), 2.hours)
    println("Done")
  }
}

