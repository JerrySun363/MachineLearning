import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesianLogisticRegression;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.MultiClassClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class ClassificationByBayesianLogisticRegression {

	/** the classifier used internally */
	protected MultiClassClassifier m_Classifier = new MultiClassClassifier();

	protected Classifier NaiveBayes_Classifier = new NaiveBayes();

	/** filtering for BayesianLogisticRegression */
	protected Filter m_Filter = null;
	protected Filter m_Filter1 = null;

	/** the training file */
	protected String m_TrainingFile = null;

	/** the training instances */
	protected Instances m_Training = null;

	/** the testing file */
	protected String m_TestingFile = null;

	/** the testing instances */
	protected Instances m_Testing = null;

	/** for evaluating the classifier */
	protected Evaluation m_Evaluation = null;
	protected Evaluation NaiveBayes_Evaluation = null;

	/** The arguments to calculate the error rate */
	protected double totalError = 0.0;
	protected double maxError = 0;
	protected int count = 0;

	public ClassificationByBayesianLogisticRegression() {
		super();
	}

	public void intializeClassifier() throws Exception {
		this.m_Classifier.setClassifier(new BayesianLogisticRegression());
	    //  this.m_Classifier=new BayesianLogisticRegression();
	}

	public void initializeFilter() throws Exception {
		m_Filter = new NominalToBinary();
		m_Filter1 = new ReplaceMissingValues();
	}

	public void initializingTraining(String name) throws Exception {
		m_TrainingFile = name;
		m_Training = new Instances(new BufferedReader(new FileReader(
				m_TrainingFile)));
		m_Training.setClassIndex(m_Training.numAttributes() - 1);
	}

	public void initializingTesting(String name) throws Exception {
		m_TestingFile = name;
		m_Testing = new Instances(new BufferedReader(new FileReader(
				m_TestingFile)));
		m_Testing.setClassIndex(m_Testing.numAttributes() - 1);
	}

	public void execute() throws Exception {
		// use m_Filter to change nomial to binary
		m_Filter.setInputFormat(m_Training);
		Instances filteredTrain = Filter.useFilter(m_Training, m_Filter);

		// use m_Filter1 to replace missing value
		m_Filter1.setInputFormat(filteredTrain);
		Instances finalFilterTrain = Filter.useFilter(filteredTrain, m_Filter1);

		// Do the same things to test data
		Instances filteredTest = Filter.useFilter(m_Testing, m_Filter);
		m_Filter1.setInputFormat(filteredTest);
		Instances finalFilterTest = Filter.useFilter(filteredTest, m_Filter1);
		int run=0;
		this.NaiveBayes_Classifier.buildClassifier(m_Training);
		NaiveBayes_Evaluation = new Evaluation(this.m_Training);
		NaiveBayes_Evaluation.evaluateModel(this.NaiveBayes_Classifier,this.m_Testing);
		BayesianLogisticRegression by = new BayesianLogisticRegression(); 
        String[] options = {"-H", "3"};
		by.setOptions(options);
        double min= Double.MAX_VALUE;
		for (double v = 0.17; v < 0.37 ;v+=0.02){
			String[] options2 = {"-V", String.valueOf(v)};
			by.setOptions(options2);
			for(double thres= 0.1; thres < 1 ;thres+=0.1){
			String[] options3 = {"-S", String.valueOf(thres)};
				by.setOptions(options3);
				m_Classifier.setClassifier(by);	
		
				double thisError = 0;
				run++;
				//System.out.println("This is run "+ run);
	     
		for (int n = 0; n < 10; n++) {
			Instances train =finalFilterTrain.trainCV(10, n);
			Instances test =finalFilterTrain.testCV(10, n);
			m_Classifier.buildClassifier(train);
			m_Evaluation = new Evaluation(train);
			m_Evaluation.evaluateModel(this.m_Classifier, test);
		
		thisError += m_Evaluation.errorRate()
				/ this.NaiveBayes_Evaluation.errorRate();
		}
		thisError/=10;
		if(min > thisError)
			min = thisError;
		
		count++;
		totalError += thisError;
		if (thisError > maxError) {
			maxError = thisError;
		}
		
		System.out.println(thisError);
		for(String s: m_Classifier.getOptions()){
			System.out.print(" "+s);
		}
		System.out.println();
		}
		}
		// Create models
		//ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(
		//		"/Users/hefuchai/Desktop/test.model"));
		//oos.writeObject(this.m_Classifier);
		//oos.flush();
		//oos.close();
		System.out.println("The dataset's min is "+ min);
	}

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		ClassificationByBayesianLogisticRegression tc = new ClassificationByBayesianLogisticRegression();
		tc.intializeClassifier();
		tc.initializeFilter();
		String[] datasets = { "anneal", "audiology", "autos", "balance-scale",
				"breast-cancer", "colic", "credit-a", "diabetes", "glass",
				"heart-c", "hepatitis", "hypothyroid" };
		for (int i = 0; i < datasets.length; i++) {
			long start = System.currentTimeMillis();
			
			tc.initializingTraining("/Users/Jerry/Documents/workspace/MachineLearning/dataout/"+datasets[i]+"_train.arff");
			tc.initializingTesting("/Users/Jerry/Documents/workspace/MachineLearning/dataout/"
					+ datasets[i] + "_test.arff");
			tc.execute();
			System.out.println("***********************");
			System.out.println("running time is "+(System.currentTimeMillis()-start)*1.0/1000);
			
			System.out.println("***********************");
			
		}
		System.out.println(tc.totalError / tc.count);
		System.out.println(tc.maxError);

	}

}
