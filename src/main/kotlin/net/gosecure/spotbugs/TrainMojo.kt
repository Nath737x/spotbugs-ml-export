package net.gosecure.spotbugs

import org.apache.maven.plugin.AbstractMojo
import org.apache.maven.plugins.annotations.Mojo
import org.apache.maven.plugins.annotations.Parameter
import org.apache.maven.project.MavenProject
import weka.classifiers.*
import net.gosecure.spotbugs.datasource.ml.MLUtils
import weka.classifiers.bayes.NaiveBayes
import weka.classifiers.functions.MultilayerPerceptron
import weka.classifiers.functions.SMO
import weka.classifiers.trees.J48
import weka.classifiers.trees.RandomForest
import weka.classifiers.trees.RandomTree

@Mojo(name="train")
class TrainMojo: AbstractMojo() {

    @Parameter(readonly = true, defaultValue = "\${project}")

    private lateinit var project: MavenProject

    private val FILE_INPUT = "aggregate-results_classified_sample.csv"
    private val FILE_OUTPUT = "aggregate-results_classified_sample.arff"
    private val MODEL_SAVED = "test-saved-model.model"

    override fun execute() {
        log.info("Training...")

        //Instantiate configuration
        val cfg = MLUtils().initConfig()

        val dataUnfiltered = MLUtils().getInstances(project, FILE_INPUT, FILE_OUTPUT)
        val dataFiltered = MLUtils().filterMeta(dataUnfiltered)
        dataFiltered.setClassIndex(dataFiltered.numAttributes() - 1)

        // Use a set of classifiers
        val models = arrayOf<Classifier>(
                NaiveBayes())

        /*val options = arrayOfNulls<String>(2)
        options[0] = "-K"
        options[1] = "4"
        (models[0] as RandomForest).setOptions(options)*/

        // Run for each model
        for (j in models.indices) {
            System.out.println("\n" + models[j].javaClass.simpleName)
            //log.info((models[j] as RandomForest).numIterations.toString())
            //log.info((models[j] as RandomForest).numFeatures.toString())

            //10 fold-cross validation, print stats data in html
            MLUtils().trainStats(project, cfg, models[j], dataFiltered)

            //Train on full data : build the classifier
            val model : Classifier = MLUtils().trainFullData(models[j], dataFiltered)

            MLUtils().saveModel(project, model, MODEL_SAVED)
        }
    }

}