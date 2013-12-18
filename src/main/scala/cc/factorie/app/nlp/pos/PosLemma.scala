package cc.factorie.app.nlp.pos
import cc.factorie._
import cc.factorie.app.nlp._
import cc.factorie.variable.StringVariable

/** Used as an attribute of Token to hold the pos lemma (siplified string, digits removed) **/
class PosLemma(val token:Token, s:String) extends StringVariable(s) {
  def lemma: String = value
  def lc: String = value.toLowerCase
}