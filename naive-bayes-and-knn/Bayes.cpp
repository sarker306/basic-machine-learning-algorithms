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
	string topic;			//document topic
	map<int,int> freqs;		//word frequency list

	document(){
	    wordCnt=0;
    }
	document(const char *str){
	    topic=string(str);
	    wordCnt=0;
    }
    
    void merge(document &doc){
        if ( doc.topic == topic ){
            for ( map<int, int>::iterator ptr = doc.freqs.begin() ; ptr != doc.freqs.end() ; ptr++ ){
                freqs[ptr->first] += ptr->second;
            }
            
            wordCnt += doc.wordCnt;
        }
    }
};

map<string,int> dict;				//global dictionary
vector<document> docs, testdocs;	//training and test documents
map<string, document>categories;
map<string, int>catCnt;
map<int, int>::iterator it;

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
//test the algo
void test();

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

    test();

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
			d.freqs[dt] = 1;		//first encounter of the word
			gfreq[dt]++;			//new word
			nwords++;
		}
		else{							//existing word in dictionary
			if(d.freqs[dt] == 0){		//new word in this document
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

    //printf("%s %d\n", d.topic.c_str(), d.wordCnt);
	return d;
}

void loadData(bool train){
    map<string, document>::iterator ptr;
    
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
			
			ptr = categories.find(d.topic);
			if ( ptr == categories.end() )
                categories.insert(pair<string, document>(d.topic, d));
            else
                ptr->second.merge(d);
                
            catCnt[d.topic]++;
		}
		else{
			testdocs.push_back(d);
			tdocs++;
		}
	}
	/*
	for ( ptr = categories.begin() ; ptr != categories.end() ; ptr++ ){
        printf("%s %d\n", ptr->first.c_str(), ptr->second.wordCnt);
    }*/
}

string findMaxProbability(document &doc){
    map<string, document>::iterator ptr;
    double best = -1e300, k = 0.005;
    string bestCat;
    
    for ( ptr = categories.begin() ; ptr != categories.end() ; ptr++ ){
        double temp = log(catCnt[ptr->first] * 1.0 / docs.size());
        
        for ( it = doc.freqs.begin() ; it != doc.freqs.end() ; it++ ){
            double num = ptr->second.freqs[it->first] + k;
            double den = ptr->second.wordCnt + k * doc.freqs.size();
            temp += log( num / den );
        }
        
        if ( best < temp ){
            best = temp;
            bestCat = ptr->first;
        }
    }
    
    return bestCat;
}

void test(){
    double accr = 0;
    for ( int i = 0 ; i < testdocs.size() ; i++ ){
        if ( findMaxProbability(testdocs[i]) == testdocs[i].topic ) accr++;
    }
    
    accr *= 100.0;
    accr /= testdocs.size();
    printf("%lf\n", accr);
}
