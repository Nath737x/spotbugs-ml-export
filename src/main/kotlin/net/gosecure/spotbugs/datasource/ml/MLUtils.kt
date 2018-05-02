package net.gosecure.spotbugs.datasource.ml

import freemarker.template.Configuration
import freemarker.template.TemplateException
import freemarker.template.TemplateExceptionHandler
import org.apache.commons.codec.digest.DigestUtils
import org.apache.maven.project.MavenProject
import org.w3c.dom.Element
import org.xml.sax.SAXException
import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.core.Instances
import weka.core.converters.ArffSaver
import weka.core.converters.CSVLoader
import weka.filters.Filter
import weka.filters.unsupervised.attribute.Add
import weka.filters.unsupervised.attribute.Remove
import java.io.*
import java.util.*
import javax.xml.parsers.DocumentBuilderFactory
import javax.xml.parsers.ParserConfigurationException
import javax.xml.transform.OutputKeys
import javax.xml.transform.TransformerException
import javax.xml.transform.TransformerFactory
import javax.xml.transform.dom.DOMSource
import javax.xml.transform.stream.StreamResult

class MLUtils {

    fun getResourcePath(project: MavenProject, fileName: String): String {

        val f = File(project.build.directory,"/spotbugs-ml")
        return f.absolutePath + File.separator + fileName
    }

    fun getResource(project: MavenProject, fileName: String): File {

        val completeFileName = getResourcePath(project, fileName)
        return File(completeFileName)
    }

    fun readFile(file:File):List<String> {
        val result = ArrayList<String>()
        val fr = FileReader(file)
        val br = BufferedReader(fr)
        var line = br.readLine()
        while (line != null)
        {
            result.add(line)
            line = br.readLine()
        }
        br.close()
        fr.close()
        return result
    }


    @Throws(IOException::class)
    fun csvToArff(csv: File, arff: File) {
        // load CSV
        val loader = CSVLoader()
        loader.setSource(csv)
        val data = loader.getDataSet()

        // save ARFF
        val saver = ArffSaver()
        saver.setInstances(data)
        saver.setFile(arff)
        saver.writeBatch()
    }

    fun getInstances(project: MavenProject, csv: String, arff: String): Instances {
        val input = getResource(project, csv)
        val output = getResource(project, arff)
        csvToArff(input, output)

        val datafile = BufferedReader(FileReader(output))

        return Instances(datafile)
    }

    fun filterMeta(data: Instances): Instances {
        var data = data
        //Filter metadatas (here SourceFile, Line, GroupId, ArtifactId, Author, key)

        val options = arrayOfNulls<String>(2)
        options[0] = "-R"
        options[1] = "1,2,3,4,5,7,15"
        val remove = Remove()
        remove.setOptions(options)
        remove.setInputFormat(data)
        data = Filter.useFilter(data, remove)

        return data
    }

    @Throws(Exception::class)
    fun createClassAttribute(data: Instances): Instances {
        var data = data
        val remove = Remove()
        remove.setAttributeIndicesArray(intArrayOf(data.numAttributes() - 1))
        remove.setInputFormat(data)
        data = Filter.useFilter(data, remove)

        val add = Add()
        add.setAttributeIndex("last")
        add.setNominalLabels("BAD,GOOD")
        add.setAttributeName("Status")
        add.setInputFormat(data)
        data = Filter.useFilter(data, add)

        return data
    }

    @Throws(Exception::class)
    fun loadModel(project: MavenProject, nameModel: String): Classifier {
        var name = project.build.directory + "/spotbugs-ml/" + nameModel
        return weka.core.SerializationHelper.read(name) as Classifier
    }

    @Throws(Exception::class)
    fun saveModel(project: MavenProject, model: Classifier, nameModel: String) {
        var name = project.build.directory + "/spotbugs-ml/" + nameModel
        weka.core.SerializationHelper.write(name, model)
    }

    @Throws(Exception::class)
    fun splitDataset(data: Instances) : Array<Instances> {
        val dataTrain = data
        val dataPredict = data

        for (i in data.numInstances() - 1..0) {
            if (data.instance(i).stringValue(data.numAttributes() - 1).equals("GOOD") ||
                    data.instance(i).stringValue(data.numAttributes() - 1).equals("BAD"))
                dataPredict.delete(i)
            else dataTrain.delete(i)
        }
        val dataSplit = arrayOf<Instances>()
        dataSplit[0] = dataTrain
        dataSplit[1] = dataPredict

        return dataSplit
    }

    @Throws(Exception::class)
    fun trainStats(project: MavenProject, cfg: Configuration, model: Classifier, data: Instances) {
        val eval = Evaluation(data)
        eval.crossValidateModel(model, data, 10, Random(1))

        val recall = java.lang.Double.toString(eval.recall(0))
        val precision = java.lang.Double.toString(eval.precision(0))
        val fmeasure = java.lang.Double.toString(eval.fMeasure(0))
        val accuracy = java.lang.Double.toString(eval.pctCorrect())
        val confusionMatrix = eval.confusionMatrix()
        val confusionMatrixString = eval.toMatrixString()

        println("Estimated Recal : " + recall)
        println("Estimated Precision : " + precision)
        println("Estimated F-measure : " + fmeasure)
        println("Estimated Accuracy : " + accuracy)
        println("Confusion Matrix : " + confusionMatrixString)

        outputHtmlTrain(project, cfg, recall, precision, fmeasure, accuracy, confusionMatrix)
    }

    @Throws(Exception::class)
    fun trainFullData(model: Classifier, data: Instances): Classifier {
        model.buildClassifier(data)
        return model
    }

    // After training, make predictions on instances, and print the prediction and real values
    @Throws(Exception::class)
    fun makePredictions(project: MavenProject, cfg: Configuration, unfiltered: Instances, unlabeled: Instances, model: Classifier, fileInput:String, fileOutput:String) {

        val labeled = Instances(unlabeled)
        val issues = ArrayList<Issue>()
        var number = 0

        val predictions = arrayOfNulls<String>(unlabeled.numInstances())
        val issuesToRemove = ArrayList<Issue>()

        for (i in 0 until (unlabeled.numInstances() - 1)) {

            val newInst = unlabeled.instance(i)

            val predNb = model.classifyInstance(newInst)
            labeled.instance(i).setClassValue(predNb)

            val predString = labeled.classAttribute().value(predNb.toInt())
            val pred = model.distributionForInstance(labeled.get(i))
            predictions[i] = predString

            //Instances classified with a probability < 90%
            if ((Math.max(pred[0], pred[1]) < 0.9) || predString.equals("GOOD")) {
                val sourceFile = unfiltered.instance(i).stringValue(0) //Source File Attribute 1
                val line = Integer.toString(unfiltered.instance(i).value(1).toInt()) //Line Attribute 2
                val bugType = unfiltered.instance(i).stringValue(5) //BugType Attribute 5
                if (Math.max(pred[0], pred[1]) < 0.9){
                    issues.add(Issue(sourceFile, line, bugType))
                    number++
                }
                if(predString.equals("GOOD")){
                    issuesToRemove.add(Issue(sourceFile, line, bugType))
                }
            }
        }
        val fileResults = project.build.directory + "/spotbugs-ml/" + fileOutput
        resultsToCsv(project, fileInput, fileResults, predictions)
        parserXml(project, issuesToRemove)
        outputHtmlPredict(project, cfg, issues, number)
    }

    fun resultsToCsv(project: MavenProject, fileInput:String, fileResults:String, predictions: Array<String?>) {
        val file = getResource(project, fileInput)
        val lines = readFile(file)
        val data = ArrayList<Array<String>>(lines.size)
        val sep = ','.toString()
        for (line in lines) {
            val oneData = line.split((sep).toRegex()).dropLastWhile({ it.isEmpty() }).toTypedArray()
            data.add(oneData)
        }
        data.removeAt(0)
        val file2 = File(fileResults)
        val fw = FileWriter(file2)
        var i = 0
        for (oneData in data) {
            fw.write(oneData[0] + "," + oneData[1] + "," + oneData[2] + "," +
                    oneData[3] + "," + oneData[4] + "," + oneData[5] + "," +
                    oneData[6] + "," + oneData[7] + "," + oneData[8] + "," +
                    oneData[9] + "," + oneData[10] + "," + oneData[11] + "," +
                    oneData[12] + "," + predictions[i] + System.getProperty("line.separator"))
            i++
        }
        fw.flush()
        fw.close()
    }

    fun sha1(str: String): String? {
        var s: String? = null
        try {
            val data = str.toByteArray(charset("UTF-8"))

            s = DigestUtils.sha1Hex(data)
        } catch (ex: Exception) {
            ex.printStackTrace()
        }

        return s
    }

    @Throws(ParserConfigurationException::class, IOException::class, SAXException::class, TransformerException::class)
    fun parserXml(project: MavenProject, issues: List<Issue>) {
        val filePath = project.build.directory + "/sonar/" +"findbugs-result.xml"
        val xmlFile = File(filePath)
        val dbFactory = DocumentBuilderFactory.newInstance()
        val dBuilder = dbFactory.newDocumentBuilder()
        val doc = dBuilder.parse(xmlFile)
        doc.documentElement.normalize()

        val nodes = doc.getElementsByTagName("BugCollection")
        val element = nodes.item(0) as Element
        val bugs = element.getElementsByTagName("BugInstance")

        //create first hashmap
        val hmap1 = HashMap<String?, Element?>()
        for (i in 0..bugs.length - 1) {
            val emp = bugs.item(i) as Element
            val newNode = emp.firstChild
            var goodNode = newNode.nextSibling
            while (!goodNode.nodeName.equals("SourceLine")){
                goodNode = goodNode.nextSibling
            }
            val sourceNode = goodNode as Element
            val sourceFile = "src/" + sourceNode.getAttribute("sourcepath")
            val line = sourceNode.getAttribute("start")
            val bugType = emp.getAttribute("type")
            hmap1.put(sha1(sourceFile + line + bugType), emp)
        }

        //create second hashmap
        val hmap2 = HashMap<String?, Issue?>()
        for (i in issues.indices) {
            hmap2.put(sha1(issues.get(i).sourceFile + issues.get(i).line + issues.get(i).bugType), issues.get(i))
        }

        //iterate through hmap2 and suppress elem in hmap1
        val it = hmap2.entries.iterator()
        while (it.hasNext()) {
            val pair = it.next() as Map.Entry<*, *>
            val elem = hmap1.get(pair.key)
            elem?.parentNode?.removeChild(elem)
        }

        doc.documentElement.normalize()
        val transformerFactory = TransformerFactory.newInstance()
        val transformer = transformerFactory.newTransformer()
        val source = DOMSource(doc)
        val result = StreamResult(File(project.build.directory + "/sonar/" +"findbugs-result-updated.xml"))
        transformer.setOutputProperty(OutputKeys.INDENT, "yes")
        transformer.transform(source, result)
        println("XML file updated successfully")

    }

    @Throws(IOException::class)
    fun initConfig(): Configuration {
        val cfg = Configuration()
        cfg.setClassForTemplateLoading(this.javaClass, "/net/gosecure/spotbugs/")
        cfg.setDefaultEncoding("UTF-8")
        cfg.setTemplateExceptionHandler(TemplateExceptionHandler.RETHROW_HANDLER)
        return cfg
    }

    @Throws(IOException::class, TemplateException::class)
    fun outputHtmlTrain(project: MavenProject, cfg: Configuration, recall: String, precision: String, fmeasure: String, accuracy: String, confusionMatrix: Array<DoubleArray>) {

        //Data model
        val map = HashMap<String, Any>()
        map.put("recall", recall)
        map.put("precision", precision)
        map.put("fmeasure", fmeasure)
        map.put("accuracy", accuracy)
        map.put("badbad", confusionMatrix[0][0])
        map.put("badgood", confusionMatrix[0][1])
        map.put("goodbad", confusionMatrix[1][0])
        map.put("goodgood", confusionMatrix[1][1])

        //Instantiate template
        val template = cfg.getTemplate("training-results.ftl")

        //Console output
        val console = OutputStreamWriter(System.out)
        template.process(map, console)
        console.flush()

        // File output
        val file = FileWriter(File(project.build.directory + "/spotbugs-ml", "training-results.html"))
        template.process(map, file)
        file.flush()
        file.close()
    }

    @Throws(IOException::class, TemplateException::class)
    fun outputHtmlPredict(project: MavenProject, cfg: Configuration, issues: List<Issue>, number: Int) {

        //Data model
        val map = HashMap<String, Any>()
        map.put("numberIssues", number.toString())
        map.put("issues", issues)

        //Instantiate template
        val template = cfg.getTemplate("predictions-results.ftl")

        //Console output
        val console = OutputStreamWriter(System.out)
        template.process(map, console)
        console.flush()

        // File output
        val file = FileWriter(File(project.build.directory + "/spotbugs-ml","predictions-results.html"))
        template.process(map, file)
        file.flush()
        file.close()
    }
}