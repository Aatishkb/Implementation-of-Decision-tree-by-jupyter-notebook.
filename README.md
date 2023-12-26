# Implementation-of-decision-tree-by-jupyter-notebook.
* Definition:-
  * Decision tree is supervised machine learning that uses labelled data and which is used to solved classification and regression problem.
  * It is a graphical representation for getting all the possible solution to a problem based on given conditions,it starts with the root node which expands on further branches and construct a tree like 
    structure where internal node is called decision node and leaf node is called output node.

* Working :-
     * Step 1:- Select the data-set
     * Step 2:- Calculate Entropy and Information Gain for each attributes
      By using following formula
i. Entropy = Entropy(s)= -P(yes)log2 P(yes)- P(no) log2 P(no)
Where,
S = Total number of samples
P(yes) = probability of yes
P(no) = probability of no
ii. Information Gain = Entropy(S)- [(Weighted Avg) *Entropy(each feature)  
     * Step 3:- Select highest information Gain for root node
