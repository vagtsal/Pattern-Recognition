// TSALESIS EVANGELOS
// AM: 1779

import java.util.Scanner;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;


public class PatternRecognition1 {
    // MAIN
    public static void main(String[] args) {
        // imports data into a string array (every element a line of data file)
        Scanner inputReader = null;
        ArrayList<String> lines = new ArrayList<>();
        try {
            inputReader = new Scanner(new FileInputStream("C:/users/vagtsal/desktop/assignment1/data/clouds.dat"));
        }
        catch (FileNotFoundException e){
            System.out.println("File not found");
            System.exit(0);
        }
        while (inputReader.hasNextLine()){
            lines.add(inputReader.nextLine());
        }
        
        System.out.println("K-Nearest Neighbour Classifier");
        kNN(lines);
        System.out.println("---------------------------");
        System.out.println("Naive Bayes Classifier");
        NBC(lines);
        System.out.println("---------------------------");
        System.out.println("Least Squares Linear Classifier");
        LS_LC(lines);
        System.out.println("---------------------------");
        System.out.println("Gradient Descent Linear Classifier");
        GD_LC(lines);
        System.out.println("---------------------------");
        
    }
    
    
    
    // **********************************'Naive Bayes classifier' method******************************************
    public static void NBC(ArrayList<String> lines){
        long time = System.currentTimeMillis();
        
        String[] sample;
        int matrixSize = lines.size()-lines.size()/10;
        double sumAccuracy = 0.0;
        
        // folds loop
        for (int f=0; f<10; f++){
            int success = 0;

            double[][] X0 = new double[matrixSize][2];
            double[][] X1 = new double[matrixSize][2];
            
            // defines matrices X0 (0 class samples), X1 (1 class samples)
            int c0 = 0;
            int c1 = 0;
            int j = 0;
            if (j == f*lines.size()/10){
                j = j + lines.size()/10;
            }
            while (j<lines.size()){
                sample  = lines.get(j).trim().split("\\s+");
                if (Byte.parseByte(sample[2]) == 0){
                    X0[c0][0] = Double.parseDouble(sample[0]);
                    X0[c0][1] = Double.parseDouble(sample[1]);
                    c0++;
                }
                else {
                    X1[c1][0] = Double.parseDouble(sample[0]);
                    X1[c1][1] = Double.parseDouble(sample[1]);
                    c1++;
                }
                
                j++;
                if (j == f*lines.size()/10){
                    j = j + lines.size()/10;
                }
            }
            
            // calculates P(class 0), P(class 1)
            double P0 = (c0+1)/(double)matrixSize;
            double P1 = (c1+1)/(double)matrixSize;
            
            // calculates means for x1|class0, x2|class0
            double sum1 = 0;
            double sum2 = 0;
            for (int i=0; i<=c0; i++){
                sum1 = sum1 + X0[i][0];
                sum2 = sum2 + X0[i][1];
            }
            double m10 = sum1 / (c0+1);
            double m20 = sum2 / (c0+1);
            
            // calculates means for x1|class1, x2|class1
            sum1 = 0;
            sum2 = 0;
            for (int i=0; i<=c1; i++){
                sum1 = sum1 + X1[i][0];
                sum2 = sum2 + X1[i][1];
            }
            double m11 = sum1 / (c1+1);
            double m21 = sum2 / (c1+1);
            
            // calculates variances for x1|class0, x2|class0
            sum1 = 0;
            sum2 = 0;
            for (int i=0; i<=c0; i++){
                sum1 = sum1 + (X0[i][0] - m10)*(X0[i][0] - m10);
                sum2 = sum2 + (X0[i][1] - m20)*(X0[i][1] - m20);
            }
            double s10 = sum1 / (c0+1);
            double s20 = sum2 / (c0+1);
            
            // calculates variances for x1|class1, x2|class1
            sum1 = 0;
            sum2 = 0;
            for (int i=0; i<=c1; i++){
                sum1 = sum1 + (X1[i][0] - m11)*(X1[i][0] - m11);
                sum2 = sum2 + (X1[i][1] - m21)*(X1[i][1] - m21);
            }
            double s11 = sum1 / (c1+1);
            double s21 = sum2 / (c1+1);
            
            // tests the estimator on the samples of the current fold
            for (int k=f*lines.size()/10; k<(f+1)*lines.size()/10; k++){
                sample = lines.get(k).trim().split("\\s+");
                double x1 = Double.parseDouble(sample[0]);
                double x2 = Double.parseDouble(sample[1]);
                byte category = Byte.parseByte(sample[2]);
                
                double normalDist10 = ((1/(Math.sqrt(2*Math.PI*s10)))*Math.pow(Math.E,-(x1-m10)*(x1-m10)/(2*s10)));
                double normalDist20 = ((1/(Math.sqrt(2*Math.PI*s20)))*Math.pow(Math.E,-(x2-m20)*(x2-m20)/(2*s20)));
                double normalDist11 = ((1/(Math.sqrt(2*Math.PI*s11)))*Math.pow(Math.E,-(x1-m11)*(x1-m11)/(2*s11)));
                double normalDist21 = ((1/(Math.sqrt(2*Math.PI*s21)))*Math.pow(Math.E,-(x2-m21)*(x2-m21)/(2*s21)));

                boolean estimator = (P0*normalDist10*normalDist20) < (P1*normalDist11*normalDist21);
                
                if ((estimator && category == 1) || ((!estimator && category == 0))){
                    success++;
                }
            }
            double accuracy = success/(lines.size()/10.00);
            sumAccuracy = sumAccuracy + accuracy;
        }    
        System.out.println("Average accuracy: " + String.format("%.3f" , sumAccuracy/10.00) + "\t\tTime: " + (System.currentTimeMillis() - time)/1000.00 + " sec");
    }   
    // ***********************************************************************************************************************

    
    
    // **********************************'Gradient-Descent Linear Classifier' method******************************************
    public static void GD_LC(ArrayList<String> lines){      
        String[] sample;
        int matrixSize = lines.size()-lines.size()/10;
        
        double hList[] = {0.001, 0.0005, 0.0001};       // learning rate
        double e = 0.0001;                              // error threshold
        
        // learning rates h loop
        for (int l=0; l<hList.length; l++){
            long time = System.currentTimeMillis();
            
            double h = hList[l];
            double sumAccuracy = 0.0;
            
            // folds loop
            for (int f=0; f<10; f++){
                int success = 0;
                double newE,oldE;
                
                double[][] X = new double[matrixSize][3];
                byte[] Y = new byte[matrixSize];
                double W[] = {0,0,0};    // initial W

                // imports xi,yi
                int i = 0;
                int j = 0;
                if (j == f*lines.size()/10){
                    j = j + lines.size()/10;
                }
                while (j<lines.size()){
                    // defines X,Y
                    sample  = lines.get(j).trim().split("\\s+");
                    X[i][0] = 1;
                    X[i][1] = Double.parseDouble(sample[0]);
                    X[i][2] = Double.parseDouble(sample[1]);
                    Y[i]    = Byte.parseByte(sample[2]);

                    i++;
                    j++;
                    if (j == f*lines.size()/10){
                        j = j + lines.size()/10;
                    }
                }

                // estimates W weights
                double sum = 0;
                for (int k=0; k<matrixSize; k++){
                    sum = sum + Math.pow((1/(1+Math.pow(Math.E, -(W[0]+ W[1]*X[k][1]+W[2]*X[k][2])))) - Y[k], 2);
                }
                newE = 0.5*sum;
                do {
                    oldE = newE;

                    sum = 0;
                    for (int m=0; m<3; m++){
                        for (int k=0; k<matrixSize; k++){
                            double phi = 1/(1+Math.pow(Math.E,-(W[0]+ W[1]*X[k][1]+W[2]*X[k][2])));
                            sum = sum + ((phi-Y[k])*phi*(1-phi)*X[k][m]);
                        }
                        W[m] = W[m] - h*sum;
                    }

                    sum = 0;
                    for (int k=0; k<matrixSize; k++){
                        sum = sum + Math.pow((1/(1+Math.pow(Math.E, -(W[0]+ W[1]*X[k][1]+W[2]*X[k][2])))) - Y[k], 2);
                    }
                    newE = 0.5*sum;
                } while(Math.abs(oldE - newE) > e);

                
                // tests the estimator on the samples of the current fold
                for (int k=f*lines.size()/10; k<(f+1)*lines.size()/10; k++){
                    sample = lines.get(k).trim().split("\\s+");
                    double x1 = Double.parseDouble(sample[0]);
                    double x2 = Double.parseDouble(sample[1]);
                    byte category = Byte.parseByte(sample[2]);
                    
                    double estimator = W[0]+ W[1]*x1+W[2]*x2;
                    if ((estimator >= 0 && category == 1) || (estimator < 0 && category == 0)){
                        success++;
                    }
                }
                double accuracy = success/(lines.size()/10.00);
                sumAccuracy = sumAccuracy + accuracy;
            }    
            System.out.println("Average accuracy for h = " + h + " : " + String.format("%.3f", sumAccuracy/10.00) + "\t\tTime: " + (System.currentTimeMillis() - time)/1000.00 + " sec");
        }
    }
    // ********************************************************************************************************************
    
    
    // **********************************'Least-Squares Linear Classifier' method******************************************
    public static void LS_LC(ArrayList<String> lines){
        long time = System.currentTimeMillis();
        
        String[] sample;
        int matrixSize = lines.size()-lines.size()/10;
        double sumAccuracy = 0.0;
        // folds loop
        for (int f=0; f<10; f++){
            int success = 0;
            
            double[][] X = new double[matrixSize][3];
            double[][] XT = new double[3][matrixSize];
            byte[] Y = new byte[matrixSize];
            
            // finds W matrix
            int i = 0;
            int j = 0;
            if (j == f*lines.size()/10){
                j = j + lines.size()/10;
            }
            while (j<lines.size()){
                // defines X,Y
                sample  = lines.get(j).trim().split("\\s+");
                X[i][0] = 1;
                X[i][1] = Double.parseDouble(sample[0]);
                X[i][2] = Double.parseDouble(sample[1]);
                if (Byte.parseByte(sample[2]) ==  0){
                    Y[i] = -1;
                }
                else{
                    Y[i] = 1;
                }

                i++;
                j++;
                if (j == f*lines.size()/10){
                    j = j + lines.size()/10;
                }
            }
            
            // calculates XT
            for (int m=0; m<matrixSize; m++){
                for (int n=0; n<3; n++){
                    XT[n][m] = X[m][n];
                }
            }
            
            // calculates XT*X
            double[][] XTX = new double[3][3];
            for (int m=0; m<3; m++){
                for (int n=0; n<3; n++){
                    double sum = 0;
                    for (int k=0; k<matrixSize; k++){
                        sum = sum + XT[m][k] * X[k][n];
                    }
                    XTX[m][n] = sum;
                }
            }
            
            // calculates inverse(XT*X)
            double det =    + XTX[0][0]*(XTX[1][1]*XTX[2][2]-XTX[1][2]*XTX[2][1]) 
                            - XTX[0][1]*(XTX[1][0]*XTX[2][2]-XTX[1][2]*XTX[2][0]) 
                            + XTX[0][2]*(XTX[1][0]*XTX[2][1]-XTX[1][1]*XTX[2][0]);   
            double[][] invXTX = new double[3][3];
            if (Math.abs(det) > 0.00001){
                invXTX[0][0] = (1.00/det)*(XTX[1][1]*XTX[2][2] - XTX[2][1]*XTX[1][2]);
                invXTX[0][1] = (1.00/det)*(XTX[0][2]*XTX[2][1] - XTX[2][2]*XTX[0][1]);
                invXTX[0][2] = (1.00/det)*(XTX[0][1]*XTX[1][2] - XTX[1][1]*XTX[0][2]);
                invXTX[1][0] = (1.00/det)*(XTX[1][2]*XTX[2][0] - XTX[2][2]*XTX[1][0]);
                invXTX[1][1] = (1.00/det)*(XTX[0][0]*XTX[2][2] - XTX[2][0]*XTX[0][2]);
                invXTX[1][2] = (1.00/det)*(XTX[0][2]*XTX[1][0] - XTX[1][2]*XTX[0][0]);
                invXTX[2][0] = (1.00/det)*(XTX[1][0]*XTX[2][1] - XTX[2][0]*XTX[1][1]);
                invXTX[2][1] = (1.00/det)*(XTX[0][1]*XTX[2][0] - XTX[2][1]*XTX[0][0]);
                invXTX[2][2] = (1.00/det)*(XTX[0][0]*XTX[1][1] - XTX[1][0]*XTX[0][1]);
            }
            else {
                System.out.println("Cannot find (XTX)-1");
                System.exit(0);
            }
            
            // calculates inverse(XT*X)*XT
            double[][] invXTX_XT = new double[3][matrixSize];
            for (int m=0; m<3; m++){
                for (int n=0; n<matrixSize; n++){
                    double sum = 0;
                    for (int k=0; k<3; k++){
                        sum = sum + invXTX[m][k] * XT[k][n];
                    }
                    invXTX_XT[m][n] = sum;
                }
            }
            
            // calculates W = inverse(XT*X)*XT*Y
            double[] W = new double[matrixSize];
            for (int m=0; m<3; m++){
                double sum = 0;
                for (int k=0; k<matrixSize; k++){
                    sum = sum + invXTX_XT[m][k] * Y[k];
                }
                W[m] = sum;
            }
            
            // tests the estimator on the samples of the current fold
            for (int k=f*lines.size()/10; k<(f+1)*lines.size()/10; k++){
                sample = lines.get(k).trim().split("\\s+");
                double x1 = Double.parseDouble(sample[0]);
                double x2 = Double.parseDouble(sample[1]);
                byte category = Byte.parseByte(sample[2]);
                
                if ((W[0] + W[1]*x1 + W[2]*x2 >= 0 && category == 1) || ((W[0] + W[1]*x1 + W[2]*x2 < 0 && category == 0))){
                    success++;
                }
            }
            double accuracy = success/(lines.size()/10.00);
            sumAccuracy = sumAccuracy + accuracy;
        }    
        System.out.println("Average accuracy : " + String.format("%.3f" , sumAccuracy/10.00) + "\t\tTime: " + (System.currentTimeMillis() - time)/1000.00 + " sec");
    }
    // ********************************************************************************************************************

    
    // **********************************'kNN-Nearest Neighbor classifier' method******************************************
    public static void kNN(ArrayList<String> lines){     
        // k loop
        for (int k=1; k<10; k=k+2){
            long time = System.currentTimeMillis();
            
            double sumAccuracy = 0.00;
            // folds loop
            for (int f=0; f<10; f++){
                int success = 0;
                // 'new' samples loop
                for (int i=f*lines.size()/10; i<(f+1)*lines.size()/10; i++){
                    int listCounter = 0;
                    NeighbourSample head = null;

                    String sample[] = lines.get(i).trim().split("\\s+");
                    double x1 = Double.parseDouble(sample[0]);
                    double x2 = Double.parseDouble(sample[1]);
                    byte category = Byte.parseByte(sample[2]);

                    // finds the neighbour samples
                    int j = 0;
                    if (j == f*lines.size()/10){
                        j = j + lines.size()/10;
                    }
                    while (j<lines.size()){
                        sample  = lines.get(j).trim().split("\\s+");
                        double xn1 = Double.parseDouble(sample[0]);
                        double xn2 = Double.parseDouble(sample[1]);
                        byte categoryn = Byte.parseByte(sample[2]);

                        double distance = Math.sqrt((x1-xn1)*(x1-xn1) + (x2-xn2)*(x2-xn2));

                        // makes a sorted linked list of k neighbour samples (keeps relative distance and category)
                        NeighbourSample newSample = new NeighbourSample();
                        newSample.distance = distance;
                        newSample.category = categoryn;
                        newSample.next = null;
                        if (head == null){
                            head = newSample;
                        }
                        else {
                            if (newSample.distance >= head.distance){
                                newSample.next = head;
                                head = newSample;
                            }
                            else{
                                NeighbourSample currentSample = head;
                                while (currentSample.next != null){
                                    if ((newSample.distance <= currentSample.distance) && (newSample.distance >= currentSample.next.distance)){
                                        newSample.next = currentSample.next;
                                        currentSample.next = newSample;
                                        break;
                                    }
                                    currentSample = currentSample.next;
                                }
                                currentSample.next = newSample; 
                            }
                        }
                        listCounter++;

                        if (listCounter > k){
                            head = head.next;
                            listCounter--;
                        } 

                        j++;
                        if (j == f*lines.size()/10){
                            j = j + lines.size()/10;
                        }
                    }

                    // sums successes and measures accuracy
                    int sum = 0;
                    NeighbourSample currentSample = head;
                    while (currentSample != null){
                        sum = sum + currentSample.category;
                        currentSample = currentSample.next;
                    }
                    if ((sum > k/2 && category == 1) || (sum <= k/2 && category == 0)){
                        success++;
                    }
                }
                double accuracy = success/(lines.size()/10.00);
                sumAccuracy = sumAccuracy + accuracy;
            }
            System.out.println("Average accuracy for k = " + k + " : " + String.format("%.3f" , sumAccuracy/10.00) + "\t\tTime: " + (System.currentTimeMillis() - time)/1000.00 + " sec");
        }
    }
}

class NeighbourSample {
    public double distance;
    public byte category;
    
    public NeighbourSample next;
}

//*************************************************************************************