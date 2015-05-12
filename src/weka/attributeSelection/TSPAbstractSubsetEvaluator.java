/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.attributeSelection;

import java.util.BitSet;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import weka.core.Instances;
import weka.core.OptionHandler;

/**
 *
 * @author diltram
 */
public abstract class TSPAbstractSubsetEvaluator extends ASEvaluation implements OptionHandler, SubsetEvaluator {

    public TSPAbstractSubsetEvaluator() {
    }

    @Override
    public abstract void buildEvaluator(Instances data) throws Exception;

    @Override
    public abstract double evaluateSubset(BitSet subset) throws Exception;

    public abstract List<BaggingRanking.ResultObject> getResultList();
    
    
    protected void sortByScore(List<BaggingRanking.ResultObject> result) {
        Collections.sort(result, new Comparator<BaggingRanking.ResultObject>() {
            @Override
            public int compare(final BaggingRanking.ResultObject object1, final BaggingRanking.ResultObject object2) {
                if (object1.getScore() != object1.getScore()) {
                    object1.setScore(-Double.MAX_VALUE);
                }
                if (object2.getScore() != object2.getScore()) {
                    object2.setScore(-Double.MAX_VALUE);
                }
                
                if (object1.getScore() < object2.getScore()) {
                    return 1;
                } else if (object1.getScore() == object2.getScore()) {
                    return 0;
                }
                return -1;
            }
        });
    }
    
}
