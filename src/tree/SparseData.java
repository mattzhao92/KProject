package tree;

import java.util.HashSet;


public class SparseData{
	public static int[][] terms_train;
	public int[][] counts_train;
	public int[][] labels_train;
	public int [] maxCount_train;
	public int[][] top_features_test;
	
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
		
		top_features_test = new int[doc_count_test][5];

		for (int i = 0; i < max_num_features; i++) {
			idf[i] = 0;
		}

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

	
	public void  compute_test_Idf(){
		for(int i = 0;  i < max_num_features; i ++){
			idf[i] ++;
			idf[i] = 1.0 / idf[i];
		}
	}
}

