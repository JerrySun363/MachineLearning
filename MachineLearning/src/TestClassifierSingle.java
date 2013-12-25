import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.ObjectOutputStream;
import java.io.Writer;
import java.net.URL;
import java.util.Scanner;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesianLogisticRegression;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.Dagging;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LADTree;
import weka.classifiers.trees.NBTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;

public class TestClassifierSingle {

	  /** the classifier used internally */
	  protected Classifier m_Classifier = null;	
	  	  
	  protected Classifier NaiveBayes_Classifier = new NaiveBayes();
	  
	  protected CVParameterSelection ps = new CVParameterSelection();
      
	  /** the training file */
	  protected String m_TrainingFile = null;

	  /** the training instances */
	  protected Instances m_Training = null;
	  
	  /** the testing file */
	  protected String m_TestingFile = null;
	  
	  /** the testing instances */	  
	  protected Instances m_Testing = null;

	  /** the model's name */
	  protected String m_model = null;
	  
	  /** for evaluating the classifier */
	  protected Evaluation m_Evaluation = null;
	  protected Evaluation NaiveBayes_Evaluation = null;
	  
	  /**variables for evaluating and calculating */
	  protected double totalError = 0.0;
	  protected double maxError= 0;
	  protected int count =0;
	  
	  protected String fileName = "";
	  
	  public TestClassifierSingle() {
		  super();
	  }
	  
	  public void intializeClassifier() throws Exception {
		      this.m_Classifier = new RandomForest();
		 
	  }
	  
	  public void initializingTraining(String name) throws Exception {
		    m_TrainingFile = name;
		    m_Training     = new Instances(new BufferedReader(new FileReader(m_TrainingFile)));
		    m_Training.setClassIndex(m_Training.numAttributes() - 1);
	  }
	  	  
	  public void initializingTesting(String name) throws Exception {
		    m_TestingFile = name;
		    m_Testing     = new Instances(new BufferedReader(new FileReader(m_TestingFile)));
		    m_Testing.setClassIndex(m_Testing.numAttributes() - 1);
	  }	  

	  
	  public void execute() throws Exception {
		
		this.m_Classifier.buildClassifier(m_Training);
		this.NaiveBayes_Classifier.buildClassifier(this.m_Training);
	    
		BufferedWriter writer1 = new BufferedWriter(new FileWriter(this.m_model));
		for(int i = 0; i < this.m_Testing.numInstances(); i++) {
		writer1.write(String.valueOf(this.m_Classifier.classifyInstance(this.m_Testing.instance(i))+"\n"));
		}
		writer1.close();
	    
	    m_Evaluation = new Evaluation(this.m_Training);
	    m_Evaluation.evaluateModel(this.m_Classifier,this.m_Testing);
	    
	    NaiveBayes_Evaluation = new Evaluation(this.m_Training);
	    NaiveBayes_Evaluation.evaluateModel(this.NaiveBayes_Classifier, this.m_Testing);
	    double thisError = m_Evaluation.errorRate()/this.NaiveBayes_Evaluation.errorRate();
	    //System.out.println(thisError);
	    System.out.println(this.fileName+"&"+this.m_Training.numAttributes()+"&"+this.m_Training.numInstances() +"&"+this.m_Training.numClasses()+"\\\\ \\hline");
		
	    
	    count++;
	    totalError +=thisError;
	    if(thisError > maxError){
	    	maxError = thisError;
	    }
		
	    
	  }
	  
	  
	@SuppressWarnings("resource")
	public static void main(String[] args) throws Exception {
		TestClassifierSingle tc = new TestClassifierSingle();
		tc.intializeClassifier();
		Scanner scanner = new Scanner(new File("/Users/Jerry/git/MachineLearning/MachineLearning/dataout/b/datafilename.txt"));
		 
		int i=0;
		while(scanner.hasNextLine()){
			i++;
			String name = scanner.nextLine();
		//String name = "sonar";
	    tc.fileName =name;
		tc.initializingTraining("/Users/Jerry/git/MachineLearning/MachineLearning/dataout/b/"+name+"_train.arff");
		tc.initializingTesting("/Users/Jerry/git/MachineLearning/MachineLearning/dataout/b/"+name+"_test.arff");
		tc.m_model = "/Users/Jerry/git/MachineLearning/MachineLearning/models/a/RF/"+name+"-LB.predict";
		long begin = System.currentTimeMillis();
	    tc.execute();
	    double end = (System.currentTimeMillis()-begin)/1000.0;
	    //System.out.println(i+ "  time is "+ end);
		}
		System.out.println(tc.totalError/tc.count);
		System.out.println(tc.maxError);
	}
}
	