#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <ctime>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <iterator>

using namespace std;

#define MAXLINESIZE 200
#define MAXDOCUMENTS 7000
#define MAXWORDS 30000
#define MAXSTORYSIZE 10000

#define PI 3.141592
#define EPS 1e-7
#define MAX(a,b) (a)>(b)?(a):(b)
#define MIN(a,b) (a)<(b)?(a):(b)

#define DELIM ",./ 0123456789\n\"\'"

#define HAM		0
#define EUCLID	1
#define TFIDF	2

#define TRAIN	true
#define TEST	false

class document{
public:
	int wordCnt;					//total words
	double tiNorm;			//TF-IDF vector norm(magnitude)
	string topic;			//document topic
	vector<int> words;		//list of word ids in this document
	map<int,double> tiw;	//TF-IDF weight
	map<int,int> freqs;		//word frequency list

	document(){
	    wordCnt=0;
	    tiNorm=0;
    }
	document(const char *str){
	    topic=string(str);
	    wordCnt=0;
	    tiNorm=0;
    }
};

map<string,int> dict;				//global dictionary
vector<document> docs, testdocs;	//training and test documents
map<int, int>::iterator it;
vector<pair<int, double> > dm;		//distance measure

bool comp(pair<int,double> a,pair<int,double> b){
	return a.second < b.second;
}

int gfreq[MAXWORDS];	//global frequency list of words
int ndocs;				//number of training documents
int tdocs;				//number of test documents
int nwords;				//number of total words

//returns a document object starting from "line" input
//determines training/testing by boolean argument "train"
document loadDoc(bool train, char *line);
//loads training/testing data
void loadData(bool train);
//test the algo for distance measure provided by 'testid' with 'ntd' test datas
void test(int testid, int ntd);

int main(int argc, char **argv){
	if ( argc < 3 ){
		printf("Usage : KNN <train file> <test file>");
		return 0;
	}

	freopen(argv[1],"r",stdin);  //redirect standard input to training data file
	loadData(TRAIN);
	printf("Total docs for training : %d\n",ndocs);
	printf("Total words             : %d\n",dict.size());

	freopen(argv[2],"r",stdin); //redirect standard input to testing data file
	loadData(TEST);
	printf("Total docs for test     : %d\n",tdocs);

    int ntd = tdocs, a2;
    
	test(HAM, ntd);
	test(EUCLID, ntd);
	test(TFIDF, ntd);

	return 0;
}

document loadDoc(bool train, char *line){
	string top(line);			//topic
	getchar();					//blank line

	gets(line);					//title
	char c = getchar();			//blank line
	if(c != '\n')
        gets(line);	//if the line wasn't blank gobble the line [wrong input]

	gets(line);					//location, date
	getchar();					//blank line

	char s[MAXSTORYSIZE]={0};	//store the story text here
	while(gets(line) != NULL){	//read story text
		if(strlen(line)==0)
			break;	//end of story text
		strcat(s,line);
		strcat(s,"\n");
	}
	if(strlen(s) == 0) return document(""); //return empty document on empty story text

	document d;

	d.topic = top;
	d.freqs.clear();

	char *p = strtok(s,DELIM);

	while(p != NULL){					//parse the story text for words
		string t(p);					//a word
		transform(t.begin(), t.end(),t.begin(), ::tolower);	//convert to lower case

		int dt = dict[t];
		if(dt == 0){					//new word in dictionary
			dt = nwords;
			dict[t] = dt;				//add the word to the dictionary
			d.words.push_back(dt);	//add the word to the documents word-list
			d.freqs[dt] = 1;		//first encounter of the word
			gfreq[dt]++;			//new word
			nwords++;
		}
		else{							//existing word in dictionary
			if(d.freqs[dt] == 0){		//new word in this document
				d.words.push_back(dt);	//add the word to word-list
				d.freqs[dt] = 1;
				if(train) gfreq[dt]++;	//?????
			}
			else{						//old word
				d.freqs[dt]++; 			//count frequency
			}
		}
		d.wordCnt++;							//count total words for the document
		p = strtok(NULL,DELIM);
	}
	
	return d;
}

void loadData(bool train){
	if(train){					//clear dictionary, global frequency list
		memset(gfreq,0,sizeof(gfreq));
		dict.clear();
	}

	char line[MAXLINESIZE];
	int i,j;

	if(train){
		ndocs = 0;
		nwords = 1;
	}
	else
	 	tdocs = 0;

	while(gets(line) != NULL){
		if(strlen(line) == 0)
			continue;	//skip blank line
		document d = loadDoc(train, line);	//load a document [train/test]
		if(d.topic.size() == 0)
			continue;	//skip the document if empty
		if(train){							//add document to specified list
			docs.push_back(d);
			ndocs++;
		}
		else{
			testdocs.push_back(d);
			tdocs++;
		}
	}

	if(train){
		for(i=0;i<ndocs;i++)
			sort(docs[i].words.begin(),docs[i].words.end());		//sort the words in the documents
	}
	else{
		for(i=0;i<tdocs;i++)
			sort(testdocs[i].words.begin(),testdocs[i].words.end());
	}
	if(train){		//calculate the TF-IDF weights for the words in the documents
		for(i=0;i<ndocs;i++){
			for(j=0;j<docs[i].words.size();j++){
				double TF = docs[i].freqs[ docs[i].words[j] ] / (docs[i].wordCnt * 1.0);
				double IDF = log((1.0*ndocs)/gfreq[ docs[i].words[j] ]);
				docs[i].tiw[ docs[i].words[j] ] = TF*IDF;
				docs[i].tiNorm += TF*IDF*TF*IDF;
				//gfreq[nwords];
			}
			docs[i].tiNorm = sqrt(docs[i].tiNorm);
		}
	}
    else{
        for(i=0;i<tdocs;i++){
            for(j=0;j<testdocs[i].words.size();j++){
                double TF = testdocs[i].freqs[ testdocs[i].words[j] ] / (testdocs[i].wordCnt * 1.0);
                double IDF = log((1.0*ndocs)/gfreq[ testdocs[i].words[j] ]);
                testdocs[i].tiw[ testdocs[i].words[j] ] = TF*IDF;
                testdocs[i].tiNorm += TF*IDF*TF*IDF;
            }
            testdocs[i].tiNorm = sqrt(testdocs[i].tiNorm);
        }
    }
}

void test(int testid, int ntd){
	int i,j,k;
	int matching[10]={0};
	register int l;
	double dist,d,a,b,acc[10];
	map<int, int>::iterator it;

	for(i=0;i<ntd;i++){		                //for all testdata
		dm.clear();				            //clear the distance measure vector
		document *testdoc = &testdocs[i];	//current test data

		for(j=0;j<ndocs;j++){	            //for all training data
			dist = 0.0;
			document *traindoc = &docs[j];	//current training data

			switch(testid){
				case(HAM):{	//Hamming Distance
					vector<int> v(testdoc->words.size()+traindoc->words.size()+2);
					set_symmetric_difference(testdoc->words.begin(),testdoc->words.end(),traindoc->words.begin(),traindoc->words.end(),v.begin());
					for(l=0;v[l];l++);
					dist += l;
					break;
				}

				case(EUCLID):{ //Euclidean Distance
					vector<int> v(traindoc->words.size()+testdoc->words.size()+2);
					set_union(traindoc->words.begin(),traindoc->words.end(),testdoc->words.begin(),testdoc->words.end(),v.begin());
					for(l=0;v[l];l++)
					{
						a=b=0;
						it = testdoc->freqs.find( v[l] );
						if(it!=testdoc->freqs.end())
							a = it->second;

						it = traindoc->freqs.find( v[l] );

						if(it!=traindoc->freqs.end())
							b = it->second;

						d = a-b;
						dist += d*d;
					}
					dist = sqrt(dist);
					break;
				}

				case(TFIDF):{ //Cosine distance
					int s = MIN(traindoc->words.size(), testdoc->words.size());
					vector<int> v(s);
					set_intersection(traindoc->words.begin(),traindoc->words.end(),testdoc->words.begin(),testdoc->words.end(),v.begin());
					for(l=0;v[l];l++)
                        dist += traindoc->tiw[ v[l] ] * testdoc->tiw[ v[l] ];
					dist /= traindoc->tiNorm;
					dist /= testdoc->tiNorm;    //now dist = cos(theta)
					dist = acos(dist);          //now dist = theta
					//printf("%.2lf\n",dist);
					break;
				}

				default:
					break;
			}
			dm.push_back(make_pair(j,dist));	//training documents with corresponding distance from current testdata
		}

		sort(dm.begin(), dm.end(), comp);		//sort thetraining documents in ascending order of distance

        map<string, int> fr;        //record topic frequencies of nearest neighbours
		for(k=1;k<=5;k+=2){         //choosing values for k
		    int max = 0, p = 0;
			for(l=0;l<k;l++){       //for k nearest neighbours in dm
                string top = docs[dm[l].first].topic;   //topic of the l'th nearest neighbour
			    fr[ top ]++ ;           //count topic frequency
			    if(fr[ top ] > max){    //record the max
                    max = fr[ top ];
                    p = dm[l].first;
			    }
			}
			if(docs[p].topic == testdoc->topic){		//matches topic!
                matching[k]++;
            }
		}
	}

    char method[40];
    if(testid == HAM)
       strcpy(method,"Hamming Distance");
    else if(testid == EUCLID)
       strcpy(method,"Euclidean Distance");
    else
       strcpy(method,"Cosine Similarity");

    printf("\n\n%-18s:\n",method);
    printf("------------------\n",method);
    for(k=1;k<=5;k+=2){
        acc[k] = (matching[k]*100.0)/ntd;
        printf("Accuracy (k=%d)    : %.2lf%%\n",k, acc[k]);
    }
}

