package tree;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;


public class SparseData{
	public static int[][] terms_train;
	public int[][] counts_train;
	public int[][] labels_train;
	public int [] maxCount_train;
	int doc_count_train;
    final static int max_num_features = 1700000;
    
    public static int[][] terms_test;
	public int[][] counts_test;
	public int [] maxCount_test;
	int doc_count_test;
	double log1 = Math.log(1);
    private double sum_sqr_idf = 0;
    
    
    
	public SparseData(int doc_count, int doc_count_test) {
		this.doc_count_train= doc_count;
		this.doc_count_test = doc_count_test;
		terms_train= new int[doc_count][];
		counts_train= new int[doc_count][];
		labels_train= new int[doc_count][];
		maxCount_train = new int[doc_count];
		maxCount_test = new int[doc_count_test];

		terms_test= new int[doc_count_test][];
		counts_test= new int[doc_count_test][];
	}
	
	public static double calculate_distance(Bundle d1, Bundle d2){
		
		//System.out.println("doc1 "+d1.doc+" doc2 "+d2.doc);
		
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
        
        int d1Size = terms_d1[d1.doc].length;
        int d2Size = terms_d2[d2.doc].length;
        int intersectSize = 0;
        
		
		for(int i = 0 ; i < terms_d1[d1.doc].length; i ++){
			//System.out.println("terms_d1 index "+i+" " +terms_d1[d1.doc][i]);
			d1Set.add(terms_d1[d1.doc][i]);
		}		
		
		for(int i = 0 ; i < terms_d2[d2.doc].length; i ++){
			//System.out.println("terms_d2 index "+i+" " +terms_d1[d2.doc][i]);
			if (d1Set.contains(terms_d2[d2.doc][i])) {
				intersectSize++;
			}
		}
		
		double result = 1 - 1.0 * intersectSize /  (d1Size + d2Size - intersectSize);
		return result;
	}	
}

