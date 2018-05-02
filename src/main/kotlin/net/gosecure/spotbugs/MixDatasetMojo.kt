package net.gosecure.spotbugs

import net.gosecure.spotbugs.datasource.ml.MLUtils
import org.apache.maven.plugin.AbstractMojo
import org.apache.maven.plugins.annotations.Mojo
import org.apache.maven.plugins.annotations.Parameter
import org.apache.maven.project.MavenProject
import weka.classifiers.Classifier
import weka.classifiers.bayes.NaiveBayes

@Mojo(name="train-predict")
class MixDatasetMojo : AbstractMojo() {

    @Parameter(readonly = true, defaultValue = "\${project}")

    private lateinit var project: MavenProject

    private val FILE_INPUT = "aggregate-results_classified_sample.csv"
    private val FILE_OUTPUT = "aggregate-results_classified_sample.arff"
    private val FILE_RESULTS = "aggregate-results_classified_sample_labeled.csv"
    private val MODEL_SAVED = "test-saved-model.model"

    override fun execute() {
        log.info("Training and predicting...")

        //Instantiate configuration
        val cfg = MLUtils().initConfig()

        val dataUnfiltered = MLUtils().getInstances(project, FILE_INPUT, FILE_OUTPUT)
        val dataFiltered = MLUtils().filterMeta(dataUnfiltered)
        dataFiltered.setClassIndex(dataFiltered.numAttributes() - 1)

        val dataSplit = MLUtils().splitDataset(dataFiltered)
        val dataTrain = dataSplit[0]
        var dataPredict = dataSplit[1]

        // Use a set of classifiers
        val models = arrayOf<Classifier>(
                NaiveBayes())

        // Run for each model
        for (j in models.indices) {
            System.out.println("\n" + models[j].javaClass.simpleName)

            //10 fold-cross validation, print stats data in html
            MLUtils().trainStats(project, cfg, models[j], dataTrain)

            //Train on full data : build the classifier
            val model : Classifier = MLUtils().trainFullData(models[j], dataTrain)
            //if needed
            MLUtils().saveModel(project, model, MODEL_SAVED)

            //Predict
            dataPredict = MLUtils().createClassAttribute(dataPredict)
            dataPredict.setClassIndex(dataPredict.numAttributes() - 1)

            MLUtils().makePredictions(project, cfg, dataUnfiltered, dataPredict, model, FILE_INPUT, FILE_RESULTS)
        }
    }
}