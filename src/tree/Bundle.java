package tree;

import java.io.Serializable;

public class Bundle implements Serializable{
     public int doc;
     public boolean isTrain;
     public Bundle(int doc, boolean isTrain){
    	 this.doc = doc;
    	 this.isTrain= isTrain;
     }
}
