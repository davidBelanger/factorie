package cc.factorie.app.nlp.pos
import cc.factorie._
import cc.factorie.app.nlp._
import cc.factorie.la._
import cc.factorie.util.{ BinarySerializer, CubbieConversions, DoubleAccumulator, FileUtils }
import java.io._
import cc.factorie.util.HyperparameterMain
import cc.factorie.variable.{ BinaryFeatureVectorVariable, CategoricalVectorDomain }
import cc.factorie.optimize.Trainer
//import cc.factorie.app.classify.backend.LinearMulticlassClassifier
import cc.factorie.app.classify.backend._

class ForwardPosTagger extends DocumentAnnotator {
  
  final val WINDOW_SIZE = 7	// should always be odd
  final val WINDOW_PAD = (WINDOW_SIZE-1)/2
  
  // Different ways to load saved parameters
  def this(stream: InputStream) = { this(); deserialize(stream) }
  def this(file: File) = this(new FileInputStream(file))
  def this(url: java.net.URL) = this(url.openConnection.getInputStream)


  // takes a window of tokens, and index to the specific token that this intance is for
  class PosTemplateInstance(val tokens: List[Token], val idx: Int, val feats: PosFeatureTemplateVariable = new PosFeatureTemplateVariable) {
    def isValidIdx(i:Int) = (i+idx) < tokens.length && (i+idx) > -1 && tokens(idx+i) != null
    def token(i:Int=0): Token = if(isValidIdx(i)) tokens(idx+i) else null
    def word(i:Int=0): String = if(isValidIdx(i)) tokens(idx+i).attr[PosLemma].value else ""
    def word_lc(i:Int=0): String = if(isValidIdx(i)) tokens(idx+i).attr[PosLemma].lc else ""
    def docFreqLemma(i:Int=0): String = if(isValidIdx(i) && WordData.docWordCounts.contains(word_lc(i))) word(i) else null
    def docFreqLemma_lc(i:Int=0): String = if(isValidIdx(i) && WordData.docWordCounts.contains(word_lc(i))) word_lc(i) else null
    def features = feats
    def label(i:Int=0): PennPosTag = if(isValidIdx(i)) tokens(idx+i).attr[PennPosTag] else null
  }

  class PosFeatureTemplateVariable extends FeatureTemplateVariable[PosTemplateInstance]

  object WordFeatureTemplate extends FeatureTemplate[PosTemplateInstance] {
    def name = "word"
    def computeFeatures(v: PosTemplateInstance, ftv: FeatureTemplateVariable[PosTemplateInstance]): Seq[String] = {
      Seq(v.word_lc())
    }
  }

  object RestFeatureTemplate extends FeatureTemplate[PosTemplateInstance] {
    def name = "rest"
    def computeFeatures(v: PosTemplateInstance, ftv: FeatureTemplateVariable[PosTemplateInstance]): Seq[String] = {

     
      //println(s"PosTemplateInstance(${v.tokens.map(x => {if(x != null) x.string else "_"}).mkString(",")})")
      //assert(v.tokens.size == WINDOW_SIZE)
      
      var feats = Seq[String]()

      // TODO cleanup (move stuff into instance)
      def lemmaStringAtOffset(offset: Int): String = "L@" + offset + "=" + v.docFreqLemma_lc(offset) // this is lowercased
      def wordStringAtOffset(offset: Int): String = "W@" + offset + "=" + v.docFreqLemma(offset) // this is not lowercased, but still has digits replaced
      def affinityTagAtOffset(offset: Int): String = "A@" + offset + "=" + WordData.ambiguityClasses.getOrElse(v.word_lc(offset), null)
      def posTagAtOffset(offset: Int): String = { val t = v.token().next(offset); "P@" + offset + (if (t ne null) t.attr[PennPosTag].categoryValue else null) }
      def takePrefix(str: String, n: Int): String = s"PREFIX=${if (n <= str.length) str.substring(0, n) else ""}"
      def takeSuffix(str: String, n: Int): String = { val l = str.length; s"SUFFIX=${if (n <= l) str.substring(l - n, l) else ""}"}
      
      // Original word, with digits replaced, no @
      val Wm2 = v.word(-2)
      val Wm1 = v.word(-1)
      val W = v.word(0)
      val Wp1 = v.word(1)
      val Wp2 = v.word(2)
      // Original words at offsets, with digits replaced, marked with @
      val wm3 = wordStringAtOffset(-3)
      val wm2 = wordStringAtOffset(-2)
      val wm1 = wordStringAtOffset(-1)
      val w0 = wordStringAtOffset(0)
      val wp1 = wordStringAtOffset(1)
      val wp2 = wordStringAtOffset(2)
      val wp3 = wordStringAtOffset(3)
      // Lemmas at offsets
      val lm2 = lemmaStringAtOffset(-2)
      val lm1 = lemmaStringAtOffset(-1)
      val l0 = lemmaStringAtOffset(0)
      val lp1 = lemmaStringAtOffset(1)
      val lp2 = lemmaStringAtOffset(2)
      // Affinity classes at next offsets
      val a0 = affinityTagAtOffset(0)
      val ap1 = affinityTagAtOffset(1)
      val ap2 = affinityTagAtOffset(2)
      val ap3 = affinityTagAtOffset(3)
      // POS tags at prev offsets
      val pm1 = posTagAtOffset(-1)
      val pm2 = posTagAtOffset(-2)
      val pm3 = posTagAtOffset(-3)
      
      feats :+= wm3
      feats :+= wm2
      feats :+= wm1
      feats :+= w0
      feats :+= wp1
      feats :+= wp2
      feats :+= wp3

      // not in ClearNLP
      //    addFeature(lp3)
      //    addFeature(lp2)
      //    addFeature(lp1)
      //    addFeature(l0)
      //    addFeature(lm1)
      //    addFeature(lm2)
      //    addFeature(lm3)

      feats :+= pm3
      feats :+= pm2
      feats :+= pm1
      feats :+= a0
      feats :+= ap1
      feats :+= ap2
      feats :+= ap3
      feats :+= lm2 + lm1
      feats :+= lm1 + l0
      feats :+= l0 + lp1
      feats :+= lp1 + lp2
      feats :+= lm1 + lp1
      feats :+= pm2 + pm1
      feats :+= ap1 + ap2
      feats :+= pm1 + ap1

      //    addFeature(pm1+a0) // Not in http://www.aclweb.org/anthology-new/P/P12/P12-2071.pdf
      //    addFeature(a0+ap1) // Not in http://www.aclweb.org/anthology-new/P/P12/P12-2071.pdf

      feats :+= lm2 + lm1 + l0
      feats :+= lm1 + l0 + lp1
      feats :+= l0 + lp1 + lp2
      feats :+= lm2 + lm1 + lp1
      feats :+= lm1 + lp1 + lp2
      feats :+= pm2 + pm1 + a0
      feats :+= pm1 + a0 + ap1
      feats :+= pm2 + pm1 + ap1
      feats :+= pm1 + ap1 + ap2

      //    addFeature(a0+ap1+ap2) // Not in http://www.aclweb.org/anthology-new/P/P12/P12-2071.pdf

      feats :+= takePrefix(W, 1)
      feats :+= takePrefix(W, 2)
      feats :+= takePrefix(W, 3)

      // not in ClearNLP
      //    addFeature("PREFIX2@1="+takePrefix(Wp1, 2))
      //    addFeature("PREFIX3@1="+takePrefix(Wp1, 3))
      //    addFeature("PREFIX2@2="+takePrefix(Wp2, 2))
      //    addFeature("PREFIX3@2="+takePrefix(Wp2, 3))

      feats :+= takeSuffix(W, 1)
      feats :+= takeSuffix(W, 2)
      feats :+= takeSuffix(W, 3)
      feats :+= takeSuffix(W, 4)

      // not in ClearNLP
      //    addFeature("SUFFIX1@1="+takeRight(Wp1, 1))
      //    addFeature("SUFFIX2@1="+takeRight(Wp1, 2))
      //    addFeature("SUFFIX3@1="+takeRight(Wp1, 3))
      //    addFeature("SUFFIX4@1="+takeRight(Wp1, 4))
      //    addFeature("SUFFIX2@2="+takeRight(Wp2, 2))
      //    addFeature("SUFFIX3@2="+takeRight(Wp2, 3))
      //    addFeature("SUFFIX4@2="+takeRight(Wp2, 4))
      feats :+= "SHAPE@-2=" + cc.factorie.app.strings.stringShape(Wm2, 2)
      feats :+= "SHAPE@-1=" + cc.factorie.app.strings.stringShape(Wm1, 2)
      feats :+= "SHAPE@0=" + cc.factorie.app.strings.stringShape(W, 2)
      feats :+= "SHAPE@1=" + cc.factorie.app.strings.stringShape(Wp1, 2)
      feats :+= "SHAPE@2=" + cc.factorie.app.strings.stringShape(Wp2, 2)
      feats :+= "HasPeriod=" + (w0.indexOf('.') >= 0)
      feats :+= "HasHyphen=" + (w0.indexOf('-') >= 0)
      feats :+= "HasDigit=" + (l0.indexOf('0', 4) >= 0) // The 4 is to skip over "W@0="
      //addFeature("MiddleHalfCap="+token.string.matches(".+1/2[A-Z].*")) // Paper says "contains 1/2+capital(s) not at the beginning".  Strange feature.  Why? -akm
      
      //println(s"features: ${feats.mkString(" ")}")     
      feats
    }
  }
  
  val templates = Seq(WordFeatureTemplate, RestFeatureTemplate)

  val domains = templates.map(t => new CategoricalFeatureTemplateDomain)
  lazy val model = new MulticlassTemplateModel[PosTemplateInstance](templates.zip(domains), PennPosDomain.size)

  /** Local lemmatizer used for POS features. */
  protected def lemmatize(string: String): String = cc.factorie.app.strings.replaceDigits(string)

  // TODO use lemmas in tokens
  /**
   * Infrastructure for building and remembering a list of training data words that nearly always have the same POS tag.
   * Used as cheap "stacked learning" features when looking-ahead to words not yet predicted by this POS tagger.
   * The key into the ambiguityClasses is app.strings.replaceDigits().toLowerCase
   */
  object WordData {
    val ambiguityClasses = collection.mutable.HashMap[String, String]()
    val sureTokens = collection.mutable.HashMap[String, Int]()
    var docWordCounts = collection.mutable.HashMap[String, Int]()
    val ambiguityClassThreshold = 0.4
    val wordInclusionThreshold = 1
    val sureTokenThreshold = -1 // -1 means don't consider any tokens "sure"

    def computeWordFormsByDocumentFrequencySentences(sentences: Iterable[Sentence], cutoff: Integer) = {
      val docSize = 1000
      var begin = 0
      for (i <- docSize.to(sentences.size).by(docSize)) {
        val docTokens = sentences.slice(begin, i).flatMap(_.tokens)
        val docUniqueLemmas = docTokens.map(x => lemmatize(x.string).toLowerCase).toSet
        for (lemma <- docUniqueLemmas) {
          if (!docWordCounts.contains(lemma)) {
            docWordCounts(lemma) = 0
          }
          docWordCounts(lemma) += 1
        }
        begin = i
      }

      // deal with last chunk of sentences
      if (begin < sentences.size) {
        val docTokens = sentences.slice(begin, sentences.size).flatMap(_.tokens)
        val docUniqueLemmas = docTokens.map(x => lemmatize(x.string).toLowerCase).toSet
        for (lemma <- docUniqueLemmas) {
          if (!docWordCounts.contains(lemma)) {
            docWordCounts(lemma) = 0
          }
          docWordCounts(lemma) += 1
        }
      }
      docWordCounts = docWordCounts.filter(_._2 > cutoff)
    }

    def computeWordFormsByDocumentFrequency(tokens: Iterable[Token], cutoff: Integer, numToksPerDoc: Int) = {
      var begin = 0
      for (i <- numToksPerDoc.to(tokens.size).by(numToksPerDoc)) {
        val docTokens = tokens.slice(begin, i)
        val docUniqueLemmas = docTokens.map(x => lemmatize(x.string).toLowerCase).toSet
        for (lemma <- docUniqueLemmas) {
          if (!docWordCounts.contains(lemma)) {
            docWordCounts(lemma) = 0
          }
          docWordCounts(lemma) += 1
        }
        begin = i
      }

      // deal with last chunk of sentences
      if (begin < tokens.size) {
        val docTokens = tokens.slice(begin, tokens.size)
        val docUniqueLemmas = docTokens.map(x => lemmatize(x.string).toLowerCase).toSet
        for (lemma <- docUniqueLemmas) {
          if (!docWordCounts.contains(lemma)) {
            docWordCounts(lemma) = 0
          }
          docWordCounts(lemma) += 1
        }
      }
      docWordCounts = docWordCounts.filter(_._2 > cutoff)
    }

    def getLemmas(tokens: Seq[Token]): (IndexedSeq[String], IndexedSeq[String]) = {
      val simplifiedForms = tokens.map(x => lemmatize(x.string))
      val lowerSimplifiedForms = simplifiedForms.map(_.toLowerCase)
      (simplifiedForms.toIndexedSeq, lowerSimplifiedForms.toIndexedSeq)
    }

    def computeAmbiguityClasses(tokens: Iterable[Token]) = {
      val posCounts = collection.mutable.HashMap[String, Array[Int]]()
      val wordCounts = collection.mutable.HashMap[String, Double]()
      var tokenCount = 0
      val lemmas = docWordCounts.keySet
      tokens.foreach(t => {
        tokenCount += 1
        if (t.attr[PennPosTag] eq null) {
          println("POS1.WordData.preProcess tokenCount " + tokenCount)
          println("POS1.WordData.preProcess token " + t.prev.string + " " + t.prev.attr)
          println("POS1.WordData.preProcess token " + t.string + " " + t.attr)
          throw new Error("Found training token with no PennPosTag.")
        }
        val lemma = lemmatize(t.string).toLowerCase
        if (!wordCounts.contains(lemma)) {
          wordCounts(lemma) = 0
          posCounts(lemma) = Array.fill(PennPosDomain.size)(0)
        }
        wordCounts(lemma) += 1
        posCounts(lemma)(t.attr[PennPosTag].intValue) += 1
      })
      lemmas.foreach(w => {
        val posFrequencies = posCounts(w).map(_ / wordCounts(w))
        val bestPosTags = posFrequencies.zip(PennPosDomain.categories).filter(_._1 > ambiguityClassThreshold).unzip._2
        val ambiguityString = bestPosTags.mkString("_")
        ambiguityClasses(w) = ambiguityString
      })
    }
  }

  var exampleSetsToPrediction = false

  class SentenceClassifierExample(val instances: Seq[PosTemplateInstance], model: MulticlassTemplateModel[PosTemplateInstance], lossAndGradient: optimize.OptimizableObjectives.Multiclass) extends optimize.Example {
    def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator) {
      instances.foreach(ins => {
        model.getFeatureVectors(ins)
        val tmpExample = new optimize.PredictorExample(model,ins.features,ins.label().intValue,lossAndGradient,1.0)
        tmpExample.accumulateValueAndGradient(value,gradient)
        if (exampleSetsToPrediction) {
          ins.token().attr[LabeledPennPosTag].set(model.classification(ins.features).bestLabelIndex)(null)
        }
      })
    }
  }

  def predict(instances: => Seq[PosTemplateInstance]): Unit = instances.foreach(ins => predict(ins))
  
  def predict(ins: PosTemplateInstance): Unit = {
    val tok = ins.token()
      if (tok.attr[PennPosTag] eq null) tok.attr += new PennPosTag(tok, "NNP")
      if (WordData.sureTokens.contains(tok.string)) {
        tok.attr[PennPosTag].set(WordData.sureTokens(tok.string))(null)
      } else {
        model.getFeatureVectors(ins)
        //println(s"ins.features.size: ${ins.features.featureVectorMap()}")
        tok.attr[PennPosTag].set(model.classification(ins.features).bestLabelIndex)(null)
      }
  }
  
  def predict(tokens: Seq[Token]): Unit = predict(generateInstances(tokens, WINDOW_PAD))
  
  def generateInstances(tokens: Seq[Token], windowSize: Int): Seq[PosTemplateInstance] = windows(tokens.toList).map(win => new PosTemplateInstance(win, WINDOW_PAD))
  
  // TODO make this generic
  def windows(l: List[Token]): List[List[Token]] = {
    (List.fill[Token](WINDOW_PAD)(null) ::: l ::: List.fill[Token](WINDOW_PAD)(null)).sliding(WINDOW_SIZE).toList
  }
    
  def predict(span: TokenSpan): Unit = predict(span.tokens)
  def predict(document: Document): Unit = {
    for (section <- document.sections)
      if (section.hasSentences) document.sentences.foreach(predict(_)) // we have Sentence boundaries 
      else predict(section.tokens) // we don't // TODO But if we have trained with Sentence boundaries, won't this hurt accuracy?
  }

  // Serialization
  def serialize(filename: String): Unit = {
    val file = new File(filename); if (file.getParentFile ne null) file.getParentFile.mkdirs()
    serialize(new java.io.FileOutputStream(file))
  }
  def deserialize(file: File): Unit = {
    require(file.exists(), "Trying to load non-existent file: '" + file)
    deserialize(new java.io.FileInputStream(file))
  }
  def serialize(stream: java.io.OutputStream): Unit = {
    import CubbieConversions._
    // TODO fix serialization
    //val sparseEvidenceWeights = new la.DenseLayeredTensor2(model.weights.value.dim1, model.weights.value.dim2, new la.SparseIndexedTensor1(_))
    //model.weights.value.foreachElement((i, v) => if (v != 0.0) sparseEvidenceWeights += (i, v))
    //model.weights.set(sparseEvidenceWeights)
    val dstream = new java.io.DataOutputStream(new BufferedOutputStream(stream))
    //BinarySerializer.serialize(FeatureDomain.domain.dimensionDomain, dstream)
    BinarySerializer.serialize(model, dstream)
    BinarySerializer.serialize(WordData.ambiguityClasses, dstream)
    BinarySerializer.serialize(WordData.sureTokens, dstream)
    BinarySerializer.serialize(WordData.docWordCounts, dstream)
    dstream.close() // TODO Are we really supposed to close here, or is that the responsibility of the caller
  }
  def deserialize(stream: java.io.InputStream): Unit = {
    import CubbieConversions._
    val dstream = new java.io.DataInputStream(new BufferedInputStream(stream))
    // TODO fix deserialization
    //BinarySerializer.deserialize(FeatureDomain.domain.dimensionDomain, dstream)
    //model.weights.set(new la.DenseLayeredTensor2(FeatureDomain.dimensionDomain.size, PennPosDomain.size, new la.SparseIndexedTensor1(_)))
    BinarySerializer.deserialize(model, dstream)
    BinarySerializer.deserialize(WordData.ambiguityClasses, dstream)
    BinarySerializer.deserialize(WordData.sureTokens, dstream)
    BinarySerializer.deserialize(WordData.docWordCounts, dstream)
    dstream.close() // TODO Are we really supposed to close here, or is that the responsibility of the caller
  }

  def printAccuracy(instancesBySentence: Seq[Seq[PosTemplateInstance]], extraText: String) = {
    var(tokAcc, senAcc, speed, _) = accuracy(instancesBySentence)
    println(extraText + s"$tokAcc token accuracy, $senAcc sentence accuracy, $speed tokens/sec")
  }
  
//  def printAccuracy(sentences: Seq[Sentence], extraText: String) = {
//    val testInstancesBySentence = sentences.map(s => generateInstances(s.tokens, WINDOW_PAD))
//    printAccuracy(testInstancesBySentence)
//  }

//  def accuracy(instances: Seq[Seq[PosTemplateInstance]]): (Double, Double) = {
//    var tokenCorrect = 0.0
//    var totalTime = 0.0
//    instances.foreach(ins => {
//      val t0 = System.currentTimeMillis()
//      predict(ins)
//      totalTime += (System.currentTimeMillis() - t0)
//      if (ins.token().attr[LabeledPennPosTag].valueIsTarget) tokenCorrect += 1.0
//    })
//    val tokensPerSecond = (instances.size / totalTime) * 1000.0
//    (tokenCorrect/instances.size, tokensPerSecond)
//  }
  
  def accuracy(instancesBySentence: Seq[Seq[PosTemplateInstance]]): (Double, Double, Double, Double) = {
    var tokenTotal = 0.0
    var tokenCorrect = 0.0
    var totalTime = 0.0
    var sentenceCorrect = 0.0
    var sentenceTotal = 0.0
    instancesBySentence.foreach(s => {
      var thisSentenceCorrect = 1.0
      val t0 = System.currentTimeMillis()
      predict(s) //predict(s)
      totalTime += (System.currentTimeMillis() - t0)
      s.foreach(ins => {
        tokenTotal += 1
        if (ins.token().attr[LabeledPennPosTag].valueIsTarget) tokenCorrect += 1.0
        else thisSentenceCorrect = 0.0
      })
      sentenceCorrect += thisSentenceCorrect
      sentenceTotal += 1.0
    })
    val tokensPerSecond = (tokenTotal / totalTime) * 1000.0
    (tokenCorrect / tokenTotal, sentenceCorrect / sentenceTotal, tokensPerSecond, tokenTotal)
  }
  
  def addPosLemmas(tokens: Seq[Token]) = tokens.foreach(tok => tok.attr += new PosLemma(tok, lemmatize(tok.string)))

  def test(sentences: Seq[Sentence]) = {
    println("Testing on " + sentences.size + " sentences...")
    // TODO re-use code
    val testTokens = sentences.flatMap(_.tokens)
    addPosLemmas(testTokens)
    val testInstancesBySentence = sentences.map(s => generateInstances(s.tokens, WINDOW_PAD))
    val (tokAcc,  speed, sentAcc, toks) = accuracy(testInstancesBySentence)
    println("Tested on " + toks + " tokens at " + speed + " tokens/sec")
    println("Token accuracy: " + tokAcc)
    println("Sentence accuracy: " + sentAcc)
  }

  def train(trainSentences: Seq[Sentence], testSentences: Seq[Sentence], lrate: Double = 0.1, decay: Double = 0.01, cutoff: Int = 2, doBootstrap: Boolean = true, useHingeLoss: Boolean = false, numIterations: Int = 5, l1Factor: Double = 0.000001, l2Factor: Double = 0.000001)(implicit random: scala.util.Random): Double = {
    // TODO Accomplish this TokenNormalization instead by calling POS3.preProcess
    //for (sentence <- trainSentences ++ testSentences; token <- sentence.tokens) cc.factorie.app.nlp.segment.PlainTokenNormalizer.processToken(token)

    val trainTokens = trainSentences.flatMap(_.tokens)
    val testTokens = testSentences.flatMap(_.tokens)
    
    addPosLemmas(trainTokens)
    addPosLemmas(testTokens)
    
    // TODO make this a parameter, also make cutoff (1 below) a parameter, and thresholds in WordData
    val toksPerDoc = 5000
    WordData.computeWordFormsByDocumentFrequency(trainTokens, 1, toksPerDoc)
    WordData.computeAmbiguityClasses(trainTokens)
    
    // how many of our lemmas had only one pos?
    // cutoff here is 1, which means this is not including words that were seen only once
    val tagCounts = WordData.ambiguityClasses.mapValues(_.count(_ == '_') + 1)
    val tagCountsWordCounts = tagCounts.zip(WordData.docWordCounts.values)
    println(s"min observed: ${tagCountsWordCounts.minBy(_._2)}; max observed: ${tagCountsWordCounts.maxBy(_._2)}")
    println(s"min tags observed: ${tagCounts.minBy(_._2)}; max tags observed: ${tagCounts.maxBy(_._2)}")
    for(i <- 2 to tagCountsWordCounts.values.max by 5)
      println(s"Percent of lemmas with one observed tag, lemma count cutoff $i: ${tagCountsWordCounts.filter(_._2 > i).count(_._1._2 == 1)/tagCounts.size.toDouble}")

    
    // Prune features by count
    //    FeatureDomain.domain.dimensionDomain.gatherCounts = true
    //    for (sentence <- trainSentences) features(sentence.tokens) // just to create and count all features
    //    FeatureDomain.domain.dimensionDomain.trimBelowCount(cutoff)
    //    FeatureDomain.domain.freeze()
    //    println("After pruning using %d features.".format(FeatureDomain.domain.dimensionDomain.size))

    
      val trainInstancesBySentence = trainSentences.map(s => generateInstances(s.tokens, WINDOW_PAD))

      trainInstancesBySentence.flatten.foreach(ins => {
        templates.zip(domains).foreach(td => {td._1.addFeatureVector(ins, ins.features, td._2)})
        //println(ins.features.featureVectorMap(WordFeatureTemplate).toString)
      })


      domains.foreach(_.freeze())
      val testInstancesBySentence = testSentences.map(s => generateInstances(s.tokens, WINDOW_PAD))
      testInstancesBySentence.flatten.foreach(ins => {
        templates.zip(domains).foreach(td => {td._1.addFeatureVector(ins, ins.features, td._2)})
        //println(ins.features.featureVectorMap(WordFeatureTemplate).toString)
      })

    templates.zip(domains).foreach(t => println(s"template ${t._1.name} domain size: ${t._2.size}"))
    
    println("finished computing features")
    
    println("Generating examples...")
    val examples = trainInstancesBySentence.shuffle.par.map(sentenceInstances =>
      new SentenceClassifierExample(sentenceInstances, model, if (useHingeLoss) cc.factorie.optimize.OptimizableObjectives.hingeMulticlass else cc.factorie.optimize.OptimizableObjectives.sparseLogMulticlass)).seq
    
    //val optimizer = new cc.factorie.optimize.AdaGrad(rate=lrate)
    val optimizer = new cc.factorie.optimize.AdaGradRDA(rate = lrate, l1 = l1Factor / examples.length, l2 = l2Factor / examples.length)
    
    //println("POS1.train\n"+trainSentences(3).tokens.map(_.string).zip(features(trainSentences(3).tokens).map(t => new FeatureVariable(t).toString)).mkString("\n"))
    def evaluate() {
      exampleSetsToPrediction = doBootstrap
      printAccuracy(trainInstancesBySentence, "Training: ")
      printAccuracy(testInstancesBySentence, "Testing: ")
      //println(s"Sparsity: ${model.weights.value.toSeq.count(_ == 0).toFloat/model.weights.value.length}")
    }
    
    println("Training...")
    Trainer.onlineTrain(model.parameters, examples, maxIterations = numIterations, optimizer = optimizer, evaluate = evaluate, useParallelTrainer = false)
    printAccuracy(trainInstancesBySentence, "Training: ")
    printAccuracy(testInstancesBySentence, "Testing: ")
    
    if (false) {
      // Print test results to file
      val source = new java.io.PrintStream(new File("pos1-test-output.txt"))
      for (s <- testSentences) {
        for (t <- s.tokens) { val p = t.attr[LabeledPennPosTag]; source.println("%s %20s  %6s %6s".format(if (p.valueIsTarget) " " else "*", t.string, p.target.categoryValue, p.categoryValue)) }
        source.println()
      }
      source.close()
    }
    accuracy(testInstancesBySentence)._1
  }

  def process(d: Document) = { predict(d); d }
  def process(s: Sentence) = { predict(s); s }
  def prereqAttrs: Iterable[Class[_]] = List(classOf[Token], classOf[Sentence], classOf[segment.PlainNormalizedTokenString])
  def postAttrs: Iterable[Class[_]] = List(classOf[PennPosTag])
  override def tokenAnnotationString(token: Token): String = { val label = token.attr[PennPosTag]; if (label ne null) label.categoryValue else "(null)" }
}

/** The default part-of-speech tagger, trained on Penn Treebank Wall Street Journal, with parameters loaded from resources in the classpath. */
class WSJForwardPosTagger(url: java.net.URL) extends ForwardPosTagger(url)
object WSJForwardPosTagger extends WSJForwardPosTagger(cc.factorie.util.ClasspathURL[WSJForwardPosTagger](".factorie"))

/** The default part-of-speech tagger, trained on all Ontonotes training data (including Wall Street Journal), with parameters loaded from resources in the classpath. */
class OntonotesForwardPosTagger(url: java.net.URL) extends ForwardPosTagger(url)
object OntonotesForwardPosTagger extends OntonotesForwardPosTagger(cc.factorie.util.ClasspathURL[OntonotesForwardPosTagger](".factorie"))

class ForwardPosOptions extends cc.factorie.util.DefaultCmdOptions with SharedNLPCmdOptions {
  val modelFile = new CmdOption("model", "", "FILENAME", "Filename for the model (saving a trained model or reading a running model.")
  val testFile = new CmdOption("testFile", "", "FILENAME", "OWPL test file.")
  val trainFile = new CmdOption("trainFile", "", "FILENAME", "OWPL training file.")
  val testDir = new CmdOption("testDir", "", "FILENAME", "Directory containing OWPL test files (.dep.pmd).")
  val trainDir = new CmdOption("trainDir", "", "FILENAME", "Directory containing OWPL training files (.dep.pmd).")
  val testFiles = new CmdOption("testFiles", "", "STRING", "comma-separated list of OWPL test files (.dep.pmd).")
  val trainFiles = new CmdOption("trainFiles", "", "STRING", "comma-separated list of OWPL training files (.dep.pmd).")
  val l1 = new CmdOption("l1", 0.000001, "FLOAT", "l1 regularization weight")
  val l2 = new CmdOption("l2", 0.00001, "FLOAT", "l2 regularization weight")
  val rate = new CmdOption("rate", 10.0, "FLOAT", "base learning rate")
  val delta = new CmdOption("delta", 100.0, "FLOAT", "learning rate decay")
  val cutoff = new CmdOption("cutoff", 2, "INT", "Discard features less frequent than this before training.")
  val updateExamples = new CmdOption("update-examples", true, "BOOL", "Whether to update examples in later iterations during training.")
  val useHingeLoss = new CmdOption("use-hinge-loss", false, "BOOL", "Whether to use hinge loss (or log loss) during training.")
  val saveModel = new CmdOption("save-model", false, "BOOL", "Whether to save the trained model.")
  val runText = new CmdOption("run", "", "FILENAME", "Plain text file on which to run.")
  val numIters = new CmdOption("num-iterations", "5", "INT", "number of passes over the data for training")
}

object ForwardPosTester {
  def main(args: Array[String]) {
    val opts = new ForwardPosOptions
    opts.parse(args)
    assert(opts.testFile.wasInvoked || opts.testDir.wasInvoked || opts.testFiles.wasInvoked)

    // load model from file if given, else use default model
    val pos = if (opts.modelFile.wasInvoked) new ForwardPosTagger(new File(opts.modelFile.value)) else OntonotesForwardPosTagger

    assert(!(opts.testDir.wasInvoked && opts.testFiles.wasInvoked))
    var testFileList = Seq(opts.testFile.value)
    if (opts.testDir.wasInvoked) {
      testFileList = FileUtils.getFileListFromDir(opts.testDir.value)
    } else if (opts.testFiles.wasInvoked) {
      testFileList = opts.testFiles.value.split(",")
    }

    val testPortionToTake = if (opts.testPortion.wasInvoked) opts.testPortion.value else 1.0
    val testDocs = testFileList.map(load.LoadOntonotes5.fromFilename(_).head)
    val testSentencesFull = testDocs.flatMap(_.sentences)
    val testSentences = testSentencesFull.take((testPortionToTake * testSentencesFull.length).floor.toInt)

    pos.test(testSentences)
  }
}

object ForwardPosTrainer extends HyperparameterMain {
  def evaluateParameters(args: Array[String]): Double = {
    implicit val random = new scala.util.Random(0)
    val opts = new ForwardPosOptions
    opts.parse(args)
    assert(opts.trainFile.wasInvoked || opts.trainDir.wasInvoked || opts.trainFiles.wasInvoked)
    // Expects three command-line arguments: a train file, a test file, and a place to save the model
    // the train and test files are supposed to be in OWPL format
    val pos = new ForwardPosTagger

    assert(!(opts.trainDir.wasInvoked && opts.trainFiles.wasInvoked))
    var trainFileList = Seq(opts.trainFile.value)
    if (opts.trainDir.wasInvoked) {
      trainFileList = FileUtils.getFileListFromDir(opts.trainDir.value)
    } else if (opts.trainFiles.wasInvoked) {
      trainFileList = opts.trainFiles.value.split(",")
    }

    assert(!(opts.testDir.wasInvoked && opts.testFiles.wasInvoked))
    var testFileList = Seq(opts.testFile.value)
    if (opts.testDir.wasInvoked) {
      testFileList = FileUtils.getFileListFromDir(opts.testDir.value)
    } else if (opts.testFiles.wasInvoked) {
      testFileList = opts.testFiles.value.split(",")
    }

    val trainDocs = trainFileList.map(load.LoadOntonotes5.fromFilename(_).head)
    val testDocs = testFileList.map(load.LoadOntonotes5.fromFilename(_).head)

    //for (d <- trainDocs) println("POS3.train 1 trainDoc.length="+d.length)
    println("Read %d training tokens from %d files.".format(trainDocs.map(_.tokenCount).sum, trainDocs.size))
    println("Read %d testing tokens from %d files.".format(testDocs.map(_.tokenCount).sum, testDocs.size))

    val trainPortionToTake = if (opts.trainPortion.wasInvoked) opts.trainPortion.value else 1.0
    val testPortionToTake = if (opts.testPortion.wasInvoked) opts.testPortion.value else 1.0
    val trainSentencesFull = trainDocs.flatMap(_.sentences)
    val trainSentences = trainSentencesFull.take((trainPortionToTake * trainSentencesFull.length).floor.toInt)
    val testSentencesFull = testDocs.flatMap(_.sentences)
    val testSentences = testSentencesFull.take((testPortionToTake * testSentencesFull.length).floor.toInt)

    pos.train(trainSentences, testSentences,
      opts.rate.value, opts.delta.value, opts.cutoff.value, opts.updateExamples.value, opts.useHingeLoss.value, numIterations = opts.numIters.value.toInt, l1Factor = opts.l1.value, l2Factor = opts.l2.value)
//    if (opts.saveModel.value) {
//      pos.serialize(opts.modelFile.value)
//      val pos2 = new ForwardPosTagger
//      pos2.deserialize(new java.io.File(opts.modelFile.value))
//      pos.printAccuracy(testDocs.flatMap(_.sentences), "pre-serialize accuracy: ")
//      pos2.printAccuracy(testDocs.flatMap(_.sentences), "post-serialize accuracy: ")
//    }
//    val acc = pos.accuracy(testDocs.flatMap(_.sentences))._1
//    if (opts.targetAccuracy.wasInvoked) cc.factorie.assertMinimalAccuracy(acc, opts.targetAccuracy.value.toDouble)
//    acc
  }
}

object ForwardPosOptimizer {
  def main(args: Array[String]) {
    val opts = new ForwardPosOptions
    opts.parse(args)
    opts.saveModel.setValue(false)
    val l1 = cc.factorie.util.HyperParameter(opts.l1, new cc.factorie.util.LogUniformDoubleSampler(1e-10, 1e2))
    val l2 = cc.factorie.util.HyperParameter(opts.l2, new cc.factorie.util.LogUniformDoubleSampler(1e-10, 1e2))
    val rate = cc.factorie.util.HyperParameter(opts.rate, new cc.factorie.util.LogUniformDoubleSampler(1e-4, 1e4))
    val delta = cc.factorie.util.HyperParameter(opts.delta, new cc.factorie.util.LogUniformDoubleSampler(1e-4, 1e4))
    val cutoff = cc.factorie.util.HyperParameter(opts.cutoff, new cc.factorie.util.SampleFromSeq(List(0, 1, 2, 3)))
    /*
    val ssh = new cc.factorie.util.SSHActorExecutor("apassos",
      Seq("avon1", "avon2"),
      "/home/apassos/canvas/factorie-test",
      "try-log/",
      "cc.factorie.app.nlp.parse.DepParser2",
      10, 5)
      */
    val qs = new cc.factorie.util.QSubExecutor(60, "cc.factorie.app.nlp.pos.ForwardPosTrainer")
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
