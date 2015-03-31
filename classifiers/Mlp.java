import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Scanner;
import java.util.zip.DataFormatException;


public class Mlp {
	
	public static double 		MOMENTUM = 0.2;
	public static double     	ERROR_RATE;
	public static double     	VALIDATION_ERROR_RATE;
	public static int        	NEURONS;
	public static int 		 	OUTPUT_NODES;
	public static boolean    	RANDOM; 
	public static double     	LEARNING_RATE;
	public static double[][] 	INPUT_LAYER;
	public static double[][] 	HIDDEN_LAYER;
	public static double[][] 	INPUT_LAYER_BACKUP;
	public static double[][] 	HIDDEN_LAYER_BACKUP;
	public static String     	TRAIN_FILE;
	public static String     	VALIDATION_FILE;
	public static String     	TEST_FILE;
	public static int 		 	PRE_VALIDATION_PERIOD; // time in which it trains befora a validation
	public static int 		 	RELEVANCE_PERIOD; // time in which the training continues after the validation error trespasses the training error
	public static int 		 	TRAIN_ERROR;
	public static int 		 	VALIDATION_ERROR;
	public static int 		 	IDEAL_EPOCH 			= -1;	
	public static StringBuffer 	VALIDATION_BUFFER      	= new StringBuffer("");
	     
	public static ArrayList<ArrayList<Double>> 	TRAIN_ANSWERS    	= new ArrayList<ArrayList<Double>>();
	public static ArrayList<ArrayList<Double>> 	VALIDATION_ANSWERS 	= new ArrayList<ArrayList<Double>>();
	public static ArrayList<ArrayList<Double>> 	TRAIN_DATA         	= new ArrayList<ArrayList<Double>>();
	public static ArrayList<ArrayList<Double>> 	VALIDATION_DATA     = new ArrayList<ArrayList<Double>>();
	public static ArrayList<ArrayList<Double>> 	TEST_DATA          	= new ArrayList<ArrayList<Double>>();
	public static ArrayList<Integer> 			PREDICTIONS 		= new ArrayList<Integer>();
	
	public static Date    		DATE 	= new Date();
	public static DateFormat 	FORMAT 	= DateFormat.getDateTimeInstance();
	

	public static void main(String[] args) throws IOException {
		//nothing much, setting up time var to use it in naming files
		String tempo =  FORMAT.format(DATE).toString();
		tempo = tempo.replaceAll("/", "-");
		tempo = tempo.replaceAll(":", "-");
		tempo = tempo.replaceAll(" ", "_");
		
		//user inputs upon java call
		TRAIN_FILE    		= args[0];
		VALIDATION_FILE 	= args[1];
		TEST_FILE     		= args[2];
		LEARNING_RATE		= Double.parseDouble(args[3]);
		NEURONS     		= Integer.parseInt(args[4]);
		RANDOM     			= Boolean.parseBoolean(args[5]);
		RELEVANCE_PERIOD 	= Integer.parseInt(args[6]);
		PRE_VALIDATION_PERIOD = Integer.parseInt(args[7]);
		
		//opens data and sets up lists 
		openData(TRAIN_DATA, TRAIN_ANSWERS, TRAIN_FILE);
		openData(VALIDATION_DATA, VALIDATION_ANSWERS, VALIDATION_FILE);
		openData(TEST_DATA, null, TEST_FILE);
		
		//used to write progress file
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("REPORT - "+TRAIN_FILE+"_"+VALIDATION_FILE+"_"+TEST_FILE+"_"+LEARNING_RATE+"_"+NEURONS+"_"+OUTPUT_NODES+"_INICIALIZACAO-ALEATORIA-"+RANDOM+"_"+RELEVANCE_PERIOD+"_"+PRE_VALIDATION_PERIOD+"_"+tempo +".txt")));
			
		//defining the architecture
		OUTPUT_NODES = 7;
		INPUT_LAYER = new double[TRAIN_DATA.get(0).size()][NEURONS];
		HIDDEN_LAYER = new double[NEURONS][OUTPUT_NODES];
			
		INPUT_LAYER_BACKUP = new double[TRAIN_DATA.get(0).size()][NEURONS];
		HIDDEN_LAYER_BACKUP = new double[NEURONS][OUTPUT_NODES];
			
		//just a litle header for the progress file
		bw.write("GIVEN PARAMETERS: \n");
		bw.write("\tTRAINING FILE: \t\t"+TRAIN_FILE+"\n");
		bw.write("\tVALIDATION FILE: \t\t\t"+VALIDATION_FILE+"\n");
		bw.write("\tTEST FILE: \t\t\t\t"+TEST_FILE+"\n");
		bw.write("\tINITIAL LEARNING RATE: \t"+LEARNING_RATE+"\n");
		bw.write("\tHIDDEN LAYER NEURONS: \t"+NEURONS+"\n");
		bw.write("\tRANDOM INITIAL WEIGHTS: \t"+RANDOM+"\n");
		bw.write("\tRELEVANCE PERIOD: \t\t\t"+RELEVANCE_PERIOD+" EPOCHS\n");
		bw.write("\tPRE-VALIDATION PERIOD: \t\t\t"+PRE_VALIDATION_PERIOD+" EPOCHS\n");
				
		//initializing weights randomly if necessary -> and writing it on the file also
		if(RANDOM){
			for(int i = 0; i<4;i++) bw.newLine();
			for(int i = 0; i<100;i++) bw.write("-");
			bw.newLine();
			bw.write("INITIAL WEIGHTS: \n");
			for(int i = 0; i<100;i++) bw.write("-");
			bw.newLine();
				
			bw.newLine();
			bw.write("INPUT LAYER: \n");
			for(int i=0;i<INPUT_LAYER.length;i++){
				bw.write("NEURON "+i+": \t");
				for(int j=0; j<INPUT_LAYER[i].length;j++){
					INPUT_LAYER[i][j] = Math.random();
					bw.write(INPUT_LAYER[i][j]+"; ");
				}
				bw.newLine();
			}
					
			bw.newLine();
			bw.write("HIDDEN LAYER: \n");
			for(int i=0;i<HIDDEN_LAYER.length;i++){
				bw.write("NEURON "+i+": \t");
				for(int j=0;j<HIDDEN_LAYER[i].length;j++){
					HIDDEN_LAYER[i][j] = Math.random();
					bw.write(HIDDEN_LAYER[i][j]+"; ");
				}
				bw.newLine();
			}
		}
				
		//preparing to start training loop.. initializing some local variables for the validation check
		int nroValidations = 0;
		int relevancePeriodCounter = 0;
		boolean activateRelevancePeriod = false;
		

		//nevermind me, just writing on the file again....
		for(int i = 0; i<4;i++) bw.newLine();
		for(int i = 0; i<100;i++) bw.write("-");
		bw.newLine();
		bw.write("TRAINING - ERROR EVOLUTION\n");
		for(int i = 0; i<100;i++) bw.write("-");
		bw.newLine();
		bw.newLine();
		bw.write("EPOCH\t\t | \t\tNUMBER OF ERRORS\t\t | \tSQUARED ERROR RATE");
		bw.newLine();
		

		int lives = 3;

		// testing something out
		/*int t = 1000;
		double previous = 0;
		boolean anchor = false;
		*/

		//training loop
		for(int i=0;i<1000000;i++){ 
			ERROR_RATE = 0;
			TRAIN_ERROR = 0;

			if(activateRelevancePeriod){
				if(relevancePeriodCounter > RELEVANCE_PERIOD){
				
					//the relevance period was active and has ended 
					// -> restore weights to backup before overfitting started
					IDEAL_EPOCH =  i;
					
					for(int k=0;k<INPUT_LAYER.length;k++){
						for(int j=0;j<INPUT_LAYER[k].length;j++){
							INPUT_LAYER[k][j] = INPUT_LAYER_BACKUP[k][j];
						}
					}
					
					for(int k=0;k<HIDDEN_LAYER.length;k++){
						for(int j=0;j<HIDDEN_LAYER[k].length;j++){
							HIDDEN_LAYER[k][j] = HIDDEN_LAYER_BACKUP[k][j];
						}
					}
					
					
					if(lives < 0) {
						//the more you know!
						System.out.println("Mlp stopped training due to overfitting...");
						break;
					} else {
						//give it another shot with a different learning rate
						LEARNING_RATE = LEARNING_RATE*0.5;
						lives--;
						activateRelevancePeriod = false;
						relevancePeriodCounter = 0;
					}					
				}
				else 
					relevancePeriodCounter++; //the relevance period is active, so count on!
			}

			//actually trains the mlp
			for(int k=0;k<TRAIN_DATA.size();k++){
				train(TRAIN_DATA.get(k),TRAIN_ANSWERS.get(k));
			}
	
			//computing the squared error rate of this epoch
			ERROR_RATE /= TRAIN_DATA.size();

			//writing on the file again
			bw.write(i + "\t\t\t\t\t" + TRAIN_ERROR + "\t\t\t\t\t\t\t" + ERROR_RATE);
			bw.newLine();

			
			//print out error every 1000 epochs to avoid flooding the terminal -> it also makes it faster :)
			if(i%1000==0){
				System.out.println("Epoch "+i+": squared error rate from TRAINING: "+ERROR_RATE);
			}
			
			//if its time for a validation check..
			if(i%PRE_VALIDATION_PERIOD==0){
				nroValidations++;
				VALIDATION_ERROR = 0;
				VALIDATION_ERROR_RATE = 0;

				//RUNS (not trains) the validation set
				for(int j=0;j<VALIDATION_DATA.size();j++){
					run(VALIDATION_DATA.get(j), VALIDATION_ANSWERS.get(j), false);
				}
				

				//computing the squared error rate of this validation
				VALIDATION_ERROR_RATE /= VALIDATION_DATA.size();
				
				//to go on the file later on..
				VALIDATION_BUFFER.append(nroValidations+ "\t\t\t\t\t" + VALIDATION_ERROR + "\t\t\t\t\t\t\t" + VALIDATION_ERROR_RATE +"\n");
				
				//print out error every 1000 epochs to avoid flooding the terminal -> it also makes it faster :)
				if(i%1000 == 0){
					System.out.println("Epoch "+i+": squared error rate from VALIDATION: "+VALIDATION_ERROR_RATE);
				}

				//if the validation error was higher than the training error AND the relevance period wasnt active
				if(VALIDATION_ERROR_RATE > ERROR_RATE && !activateRelevancePeriod){
					
					//it means the mlp may be starting to overfit, so backup the weights just to be sure
					for(int k=0;k<INPUT_LAYER.length;k++){
						for(int j=0;j<INPUT_LAYER[k].length;j++){
							INPUT_LAYER_BACKUP[k][j] = INPUT_LAYER[k][j];
						}
					}
					
					for(int k=0;k<HIDDEN_LAYER.length;k++){
						for(int j=0;j<HIDDEN_LAYER[k].length;j++){
							HIDDEN_LAYER_BACKUP[k][j] = HIDDEN_LAYER[k][j];
						}
					}
					
					//set on the flags
					activateRelevancePeriod = true;
					relevancePeriodCounter = 0;
	
				}
			
				// if the relevance period WAS active and the validation error went below the training
				// then the overfitting alert was false alarm, keep going
				if(activateRelevancePeriod && VALIDATION_ERROR_RATE<=ERROR_RATE ){
					activateRelevancePeriod = false;
					relevancePeriodCounter = 0;
				}
			}
				

			// that thing i was testing 
			/*if(intervallContains(ERROR_RATE-(ERROR_RATE*0.001), ERROR_RATE+(ERROR_RATE*0.001), previous)){
				anchor = true;
				IDEAL_EPOCH = i;
			} else {
				anchor = false;
				t = 1000;
			}

			if(anchor && t--<=0){

				System.out.println("Stopped with current error of "+ERROR_RATE+" and previous of "+previous);
				break;
			}

			previous = ERROR_RATE;
			*/
		}

		//the more you know!
		System.out.println("Training has ended, starting to run test...");
		
		//actually runs the test
		for(int k=0;k<TEST_DATA.size();k++){
			run(TEST_DATA.get(k), null, true);
		}
		
		//on the file again!
		for(int i = 0; i<4;i++) bw.newLine();
		for(int i = 0; i<100;i++) bw.write("-");
		bw.newLine();
		bw.write("VALIDATION - ERROR EVOLUTION\n");
		for(int i = 0; i<100;i++) bw.write("-");
		bw.newLine();
		bw.newLine();
		bw.write("ID\t\t\t | \t\tNUMBER OF ERRORS\t\t | \t\tSQUARED ERROR RATE \n");
		bw.write(VALIDATION_BUFFER.toString());
		bw.newLine();
		for(int i = 0; i<4;i++) bw.newLine();
		for(int i = 0; i<100;i++) bw.write("-");
		bw.newLine();
		bw.write("FINAL CONFIGURATIONS:\n");
		for(int i = 0; i<100;i++) bw.write("-");
		bw.newLine();
		
		bw.write("IDEAL CONFIGURATION FOUND AT EPOCH: "+IDEAL_EPOCH+"\n");
		bw.write("FINAL RATE: "+LEARNING_RATE+"\n");
		bw.write("FINAL WEIGHTS: \n");
		bw.newLine();
		bw.write("INPUT LAYER: \n");
		for(int i=0;i<INPUT_LAYER.length;i++){
			bw.write("NEURON "+i+": \t");
			for(int j=0; j<INPUT_LAYER[i].length;j++){
				bw.write(INPUT_LAYER[i][j]+"; ");
			}
			bw.newLine();
		}
		
		bw.newLine();
		bw.write("HIDDEN LAYER: \n");
		for(int i=0;i<HIDDEN_LAYER.length;i++){
			bw.write("NEURON "+i+": \t");
			for(int j=0;j<HIDDEN_LAYER[i].length;j++){
				bw.write(HIDDEN_LAYER[i][j]+"; ");
			}
			bw.newLine();
		}
		
		bw.flush();
		bw.close();



		//write predictions on the csv file
		int answer_id = 15121;
		bw = new BufferedWriter(new FileWriter(new File("Predictions_"+TRAIN_FILE+"_"+VALIDATION_FILE+"_"+TEST_FILE+"_"+LEARNING_RATE+"_"+NEURONS+"_"+OUTPUT_NODES+"_INICIALIZACAO-ALEATORIA-"+RANDOM+"_"+RELEVANCE_PERIOD+"_"+PRE_VALIDATION_PERIOD+"_"+tempo +".csv")));
		bw.write("Id,Cover_Type");
		bw.newLine();
		for(int a:PREDICTIONS){
			bw.write(answer_id + ","+ a);
			bw.newLine();
			answer_id+= 1;
		}
		bw.flush();
		bw.close();
	}
	

	//feedforward and backpropagation
	public static void train(ArrayList<Double> input, ArrayList<Double> expect){
		double[]   z_in            			= new double[HIDDEN_LAYER.length];
		double[]   z               			= new double[HIDDEN_LAYER.length];
		double[]   y_in            			= new double[OUTPUT_NODES];
		double[]   y               			= new double[OUTPUT_NODES];
		double[]   erro_y_in       			= new double[OUTPUT_NODES];
		double[][] correct_HIDDEN_LAYER 	= new double[HIDDEN_LAYER.length][OUTPUT_NODES];
		double[]   erro_z_in       			= new double[HIDDEN_LAYER.length];
		double[]   erro_z		  			= new double[HIDDEN_LAYER.length];
		double[][] correct_INPUT_LAYER		= new double[INPUT_LAYER.length][INPUT_LAYER[0].length];
		
		//feedforward starts...
		for(int k=0;k<INPUT_LAYER.length;k++){ 
			for(int i=0;i<INPUT_LAYER[k].length;i++){
				z_in[i] += INPUT_LAYER[k][i] * input.get(k);
			}
		}

		for(int i=0;i<z_in.length;i++) 
			z[i] = Sigmoid(z_in[i]); 
		
		for(int i=0;i<y_in.length;i++){
			for(int k=0;k<z.length;k++){
				y_in[i] += z[k]*HIDDEN_LAYER[k][i]; 
			}
		}
		
		for(int i=0;i<y_in.length;i++)
			y[i] = Sigmoid(y_in[i]);
		
		//reached the output.. compare with the answer to compute the error
		for(int i=0;i<y_in.length;i++)
			erro_y_in[i] = (expect.get(i) - y[i]) * Derivate(y_in[i]);
		

		//used to find out the answer it gave
		double highest = -99.0;
		double highestExpect = -99.0;
		int indexM = -1;
		int indexExpect = -1;
		
		for(int i=0;i<OUTPUT_NODES;i++){
			if(highestExpect < expect.get(i)){
				highestExpect = expect.get(i);
				indexExpect = i;
			}
			
			if(highest < y[i]){
				highest = y[i];
				indexM = i;
			}			
		}
		
		//indexM contains the actual prediction and indexExpect, the right answer
		if(indexM != indexExpect) TRAIN_ERROR++;
				

		double localError = 0;
		for(int i=0;i<y_in.length;i++)
			localError += Math.abs(expect.get(i) - y[i]);
		
		localError *= localError; //squared error
		
		ERROR_RATE += localError/OUTPUT_NODES; //squared error rate

		//backpropagation..
		for(int k=0;k<correct_HIDDEN_LAYER.length;k++){
			for(int i=0;i<correct_HIDDEN_LAYER[k].length;i++){
				correct_HIDDEN_LAYER[k][i] = LEARNING_RATE * erro_y_in[i] * z[k];
			}
		}
		
		for(int i=0;i<erro_z_in.length;i++){
			for(int k=0;k<OUTPUT_NODES;k++){
				erro_z_in[i] += erro_y_in[k] * HIDDEN_LAYER[i][k];
			}
		}
		
		for(int i=0;i<erro_z.length;i++) 
			erro_z[i] = erro_z_in[i] * Derivate(z_in[i]);
		
		for(int k=0;k<correct_INPUT_LAYER.length;k++){ 
			for(int i=0;i<correct_INPUT_LAYER[k].length;i++){
				correct_INPUT_LAYER[k][i] = LEARNING_RATE * erro_z[i] * input.get(k);
			}
		}
		
		for(int k=0;k<correct_HIDDEN_LAYER.length;k++){
			for(int i=0;i<correct_HIDDEN_LAYER[k].length;i++){
				HIDDEN_LAYER[k][i] = HIDDEN_LAYER[k][i] + (1+MOMENTUM)*correct_HIDDEN_LAYER[k][i];
			}
		}
		
		for(int k=0;k<correct_INPUT_LAYER.length;k++){
			for(int i=0;i<correct_INPUT_LAYER[k].length;i++){
				INPUT_LAYER[k][i] = INPUT_LAYER[k][i] + (1+MOMENTUM)*correct_INPUT_LAYER[k][i];
			}
		}
	}
	
	//feedforward
	public static int run(ArrayList<Double> input, ArrayList<Double> expect, boolean test){
		double[]  z_in 	= new double[HIDDEN_LAYER.length];
		double[]  z     = new double[HIDDEN_LAYER.length];
		double[]  y_in  = new double[OUTPUT_NODES];
		double[]  y     = new double[OUTPUT_NODES];
		
		for(int k=0;k<INPUT_LAYER.length;k++){ 
			for(int i=0;i<INPUT_LAYER[k].length;i++){
				z_in[i] += INPUT_LAYER[k][i] * input.get(k);
			}
		}
		
		for(int i=0;i<z_in.length;i++) 
			z[i] = Sigmoid(z_in[i]);
		
		for(int i=0;i<y_in.length;i++){
			for(int k=0;k<z.length;k++){
				y_in[i] += z[k]*HIDDEN_LAYER[k][i]; 
			}
		}
		
		for(int i=0;i<y_in.length;i++)
			y[i] = Sigmoid(y_in[i]); 
			
		
		//used to find out the answer it gave
		double highest = -99.0;
		double highestExpect = -99.0;
		int indexM = -1;
		int indexExpect = -1;
		
		for(int i=0;i<OUTPUT_NODES;i++){
			//if is not the test then we have a answer to compare to
			if(!test){
				if(highestExpect < expect.get(i)){
					highestExpect = expect.get(i);
					indexExpect = i;
				}
			}
			if(highest < y[i]){
				highest = y[i];
				indexM = i;
			}
			
		}
		
		//if is not the test then we have a answer to compare to
		if(!test){
			//indexM contains the actual prediction and indexExpect, the right answer
			if(indexM != indexExpect) {
				VALIDATION_ERROR++;

				double erro_local = 0;
				for(int i=0;i<y_in.length;i++)
					erro_local += Math.abs(expect.get(i) - y[i]);
					
				erro_local *= erro_local;

				VALIDATION_ERROR_RATE += erro_local/OUTPUT_NODES;
			}
		}else {
			//if it is the test, write the prediction
			PREDICTIONS.add(indexM+1);
		}
		return indexM+1;
	}
	
	//auxiliary functions
	public static double Sigmoid(double num){
		return (1/(1+ Math.exp(-num)));
	}
	
	public static double Derivate(double num){
		return(Sigmoid(num) * (1 - Sigmoid(num)));
	}

	public static boolean intervallContains(double low, double high, double n) {
    	return n >= low && n <= high;
	}
	
	//opens the data 
	public static void openData(ArrayList<ArrayList<Double>> input, 
		ArrayList<ArrayList<Double>> answers, String fileName) throws IOException{
		
		//x = 1 if there are answers to compare to, x = 0 if there arent
		int x = 1;
		if(answers == null){
			x = 0;
		}

		BufferedReader bf = new BufferedReader(new FileReader(new File(fileName)));
		
		String entry = bf.readLine();//ignores the header line
		
		while((entry = bf.readLine()) != null){
			String[] data = entry.split(",");

			ArrayList<Double> aux = new ArrayList<Double>();
			for(int i=1;i<data.length -x;i++){	//ignores the id column
				aux.add(Double.parseDouble(data[i]));
			}
						
			//if it was a test data, there is nothing to compare to, move along
			if(answers == null) {
				input.add(aux);
				continue;
			}
			//otherwise, add the right answers to the expected list
			if(data[data.length -1].equals("1")){ 
				ArrayList<Double> expected = new ArrayList<Double>();
				expected.add(1.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
					
				answers.add(expected);
			}
			else if(data[data.length -1].equals("2")){ 
				ArrayList<Double> expected = new ArrayList<Double>();
				expected.add(0.0);
				expected.add(1.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);

				answers.add(expected);
			}
			if(data[data.length -1].equals("3")){ 
				ArrayList<Double> expected = new ArrayList<Double>();
				expected.add(0.0);
				expected.add(0.0);
				expected.add(1.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);

				answers.add(expected);
			}
				
			if(data[data.length -1].equals("4")){ 
				ArrayList<Double> expected = new ArrayList<Double>();
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(1.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);

				answers.add(expected);
			}
			if(data[data.length -1].equals("5")){ 
				ArrayList<Double> expected = new ArrayList<Double>();
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(1.0);
				expected.add(0.0);
				expected.add(0.0);

				answers.add(expected);
			}
			if(data[data.length -1].equals("6")){ 
				ArrayList<Double> expected = new ArrayList<Double>();
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(1.0);
				expected.add(0.0);

				answers.add(expected);
			}
			if(data[data.length -1].equals("7")){ 
				ArrayList<Double> expected = new ArrayList<Double>();
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(0.0);
				expected.add(1.0);

				answers.add(expected);
			}
			
			input.add(aux);
		}
	}
}
	
