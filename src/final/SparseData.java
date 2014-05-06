
import java.io.*;
import java.util.HashMap;
import java.util.HashSet;



public class SparseData{
	public static int[][] terms_train;
	public int[][] counts_train;
	public int[][] labels_train;
	public int [] maxCount_train;
	public int[][] top_features_test;
	public int[][] five_neighbors;
	public HashMap<Integer,Integer> label_count ;
	public HashMap<Integer,Integer> last_layer;
    private double sum_sqr_idf = 0;

	
	
	int doc_count_train;
    final static int max_num_features = 1700000;
    
    public static int[][] terms_test;
	public int[][] counts_test;
	public int [] maxCount_test;
	int doc_count_test;
	double log1 = Math.log(1);
    
    
    
	public SparseData(int doc_count, int doc_count_test) {
		this.doc_count_train= doc_count;
		this.doc_count_test = doc_count_test;
		terms_train= new int[doc_count][];
		counts_train= new int[doc_count][];
		labels_train= new int[doc_count][];		
		terms_test= new int[doc_count_test][];
		counts_test= new int[doc_count_test][];
		idf = new double[max_num_features];
		five_neighbors = new int[doc_count_test][20];
		label_count = new HashMap<Integer,Integer>();
		last_layer = new HashMap<Integer,Integer>();

		top_features_test = new int[doc_count_test][5];

		for (int i = 0; i < max_num_features; i++) {
			idf[i] = 0;
		}
	}
	
	public void load_label_count() throws Exception{
		BufferedReader br = new BufferedReader(new FileReader("labelCount.txt"));
		String line;
		while ((line = br.readLine()) != null) {
		   // process the line.
			String[] arr = line.split(" ");
			label_count.put(Integer.parseInt(arr[0]), Integer.parseInt(arr[1]));
		}
		System.out.println("Finished loading label count");
		br.close();
	}
	
	public void load_last_layer() throws Exception{
		BufferedReader br = new BufferedReader(new FileReader("lastLayer.txt"));
		String line;
		while ((line = br.readLine()) != null) {
		   // process the line.
			String[] arr = line.split(" ");
			last_layer.put(Integer.parseInt(arr[0]), Integer.parseInt(arr[1]));
		}
		System.out.println("Finished loading last layer");
		br.close();
	}

	
	public void load_knn(String uri, int num) throws Exception{
		System.out.println("starting loading knn");
		BufferedReader br = new BufferedReader(new FileReader(uri));
		String line;
		while ((line = br.readLine()) != null) {
		   // process the line.
			String[] arr = line.split(" ");
			int index = Integer.parseInt(arr[0]);
			for(int i = 0; i < Math.min(num, arr.length-1) ; i ++){
				five_neighbors[index][i] = Integer.parseInt(arr[i+1]);
			}
		}
		System.out.println("Finished loading knn");
		br.close();
	}
	
public double cosine_similarity(Bundle d1, Bundle d2){
		
		System.out.println("doc1 "+d1.doc+" doc2 "+d2.doc);
		
		int[][] terms_d1;
		int[][] terms_d2;

        if(d1.isTrain) {
        	terms_d1 = terms_train;
        }else{
        	terms_d1 = terms_test;
        }
        
        if(d2.isTrain) {
        	terms_d2 = terms_train;
        }else{
        	terms_d2 = terms_test;
        }
		HashSet<Integer> d1Set = new HashSet<Integer>();
		for(int i = 0 ; i < terms_d1[d1.doc].length; i ++){
			//System.out.println("terms_d1 index "+i+" " +terms_d1[d1.doc][i]);
			d1Set.add(terms_d1[d1.doc][i]);
		}		
		HashSet<Integer> d2Set = new HashSet<Integer>();
		for(int i = 0 ; i < terms_d2[d2.doc].length; i ++){
			//System.out.println("terms_d2 index "+i+" " +terms_d1[d2.doc][i]);
			d2Set.add(terms_d2[d2.doc][i]);
		}	
		HashSet<Integer> intersect = new HashSet<Integer>(d1Set);
		intersect.retainAll(d2Set);
		
		HashSet<Integer> d1Diff = new HashSet<Integer>(d1Set);
		d1Diff.removeAll(intersect);
		HashSet<Integer> d2Diff = new HashSet<Integer>(d2Set);
		d2Diff.removeAll(intersect);
		
		
		double numerator = 0;
		double temp_sum_sqr_idf = sum_sqr_idf;
		
		for(Integer term : intersect  ){
			numerator += gettfidf(term,d1) * gettfidf(term,d2);
			temp_sum_sqr_idf  -= idf[term] * idf[term];

		}
		for(Integer term : d1Diff  ){
			numerator += gettfidf(term,d1) * 0.5 * idf[term] ;
			temp_sum_sqr_idf  -= idf[term] * idf[term];


		}
		for(Integer term : d2Diff  ){
			numerator += gettfidf(term,d2) * 0.5 * idf[term] ;
			temp_sum_sqr_idf  -= idf[term] * idf[term];

		}
		
		numerator += 0.25*temp_sum_sqr_idf;				
		double divisor = 0;
		double sum1 = 0;
		double sumsqridf1 = sum_sqr_idf;
		for(Integer term : d1Set){
			double temptfidf = gettfidf(term,d1);
			//System.out.println("term 1 "+term);

			sum1 += temptfidf * temptfidf;
			sumsqridf1 -= idf[term]*idf[term];
		}
		sum1 += 0.25 * sumsqridf1;	
		double sum2 = 0;
		double sumsqridf2 = sum_sqr_idf;
		for(Integer term : d2Set){
			double temptfidf = gettfidf(term,d2);
			//System.out.println("term 2 "+term);
			sum2 += temptfidf * temptfidf;
			sumsqridf2 -= idf[term]*idf[term];
		}
		sum2 += 0.25 * sumsqridf2;
		divisor  = Math.sqrt(sum1) * Math.sqrt(sum2);
		
		
		double result = numerator / divisor;
		//System.out.println("cosine smilarity: sum1 "+sum1+" sum2 "+sum2+" sumsqridf1 "+sumsqridf1+
		//		" sumsqridf2 "+sumsqridf2+""+ "numerator is "+ numerator +" divisor is "+divisor);
		 System.out.println("cosine similarity is "+result);
		
		 /*
		double result2 = 0;
		double sum1_brute = 0;
		double sum2_brute = 0;
		for (int i = 0; i < max_num_features; i++) {
			double temptfidf = gettfidf(i, d1);
			sum1_brute += temptfidf * temptfidf;
		}
		
		for (int i = 0; i < max_num_features; i++) {
			double temptfidf = gettfidf(i, d2);
			sum2_brute += temptfidf * temptfidf;
		}
		
		 System.out.println("cosine smilarity: sum1 = "+sum1+" sum2 = "+sum2+
				" brute sum1 = "+ sum1_brute+ " brute sum2 = "+sum2_brute+
				" sumsqridf1 "+sumsqridf1+
			" sumsqridf2 "+sumsqridf2+""+ "numerator is "+ numerator +" divisor is "+divisor);
		*/
		return result;	
	}
	
	
	public static double calculate_distance(Bundle d1, Bundle d2){		
		int[][] terms_d1, terms_d2;
        
		if(d1.isTrain) {
        	terms_d1 = terms_train;
        }else{
        	terms_d1 = terms_test;
        }
        
        if(d2.isTrain) {
        	terms_d2 = terms_train;
        }else{
        	terms_d2 = terms_test;
        }
       
        HashSet<Integer> d1Set = new HashSet<Integer>();
        
        int d1Size = terms_d1[d1.doc].length;
        int d2Size = terms_d2[d2.doc].length;
        int intersectSize = 0;
		
		for(int i = 0 ; i < terms_d1[d1.doc].length; i ++) {
			d1Set.add(terms_d1[d1.doc][i]);
		}		
		
		for(int i = 0 ; i < terms_d2[d2.doc].length; i ++) {
			if (d1Set.contains(terms_d2[d2.doc][i])) {
				intersectSize++;
			}
		}
		
		return 1 - 1.0 * intersectSize /  (d1Size + d2Size - intersectSize);
	}	
	
	
	

	public double[] idf;

    
	// first param: term second param: doc
	public double gettfidf (int t, Bundle d){
		
		if(d.isTrain){
			for( int i = 0 ; i < terms_train[d.doc].length; i ++){
				if( terms_train[d.doc][i] == t){
					//System.out.println("max count train "+maxCount_train[d.doc]);
					return (counts_train[d.doc][i]+1)*idf[t];
				}
			}
			return idf[t];
			//throw new RuntimeException("illegal t");
		}
		else{

			for( int i = 0 ; i < terms_test[d.doc].length; i ++){
				if( terms_test[d.doc][i] == t){
					return (counts_test[d.doc][i]+1)*idf[t];
				}
			}
			return idf[t];
		}
	}
	
	
	public void addCount(int term){
		idf[term] ++;
	}

	/*
	public void  compute_test_Idf(){
		for(int i = 0;  i < max_num_features; i ++){
			idf[i] ++;
			idf[i] = 1.0 / idf[i];
		}
	}
    */	
	
	public void  compute_test_Idf(){
		for(int i = 0;  i < max_num_features; i ++){
			idf[i] ++;
			//if(idf[i] !=0){
			idf[i] = Math.log(1.0*doc_count_test / idf[i]);
			//}
			//System.out.println("test idf is "+idf[i]);
			sum_sqr_idf += idf[i]*idf[i];
			
		}
	}
}

