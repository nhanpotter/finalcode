# coding=utf-8
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import numpy as np
import gc

def train_word2vec(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r', encoding='utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model
    
def create_embedding_matrix(tokenizer, word_vectors, embedding_dim):
    nb_words = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((nb_words, embedding_dim))
    ## Handling exceptions in the dataset, specifically the reference answer and student answer fields
    for word, i in word_index.items():
        if(word == '*'):
            embedding_vector = np.zeros(300)
        elif(word == 'itç\x90ç¢ç_\x90_çâç_\x90_ç¢s'):
            embedding_vector = np.zeros(300)
        elif(word == 'it\x82\x90\x82¢\x82_\x90_\x82\x89\x82_\x90_\x82¢s'):
            embedding_vector = np.zeros(300)
        elif(word == 'array\x82\x90\x90_\x90_\x82¢\x82\x90\x82¢\x90_\x82\x89\x82\x90\x82¢\x90_\x82¢s'):
            embedding_vector = word_vectors['array']
        elif(word == '*f'):
            embedding_vector = word_vectors['f']
        elif(word == '\x82\x90\x82¢\x82_\x90_\x82\x89\x82\x81\x82_do'):
            embedding_vector = np.zeros(300)
        elif(word == '\x82\x90\x82¢\x82_\x90_\x82\x89\x82\x81\x82_do\x82\x90\x82¢\x82_\x90_\x82\x89\x90_\x82\x9d'):
            embedding_vector = np.zeros(300)
        elif(word == 'itâ\x90â¢â_\x90_âäâ_\x90_â¢s'):
            embedding_vector = word_vectors['it']
        elif(word == 'while\x82\x90\x82¢\x82_\x90_\x82\x89\x90_\x82\x9d'):
            embedding_vector = word_vectors['while']
        elif(word == '\x82\x90\x82¢\x82_\x90_\x82\x89\x82_\x82_\x82\x9d'):
            embedding_vector = np.zeros(300)
        elif(word == '\x82\x90\x82¢\x82_\x90_\x82\x89\x82\x8f\x82_int'):
            embedding_vector = np.zeros(300)
        elif(word == '\x82\x90\x82¢\x82_\x90_\x82\x89\x82_\x90_\x82¢'):
            embedding_vector = np.zeros(300)
        elif(word == 'object\x82\x90\x82¢\x82_\x90_\x82\x89\x82_\x90_\x82¢s'):
            embedding_vector = word_vectors['object']
        elif(word == 'definition\x82\x90\x82¢\x82_\x90_\x82\x89\x82_\x90_\x82¢s'):
            embedding_vector = word_vectors['definition']
        elif(word == 'arrayç\x90\x90_\x90_ç¢ç\x90ç¢\x90_çâç\x90ç¢\x90_ç¢s'):
            embedding_vector = word_vectors['array']
        elif(word == 'ç\x90ç¢ç_\x90_çâç\x81ç_frontç\x90ç¢ç_\x90_çâ\x90_ç\x9d'):
            embedding_vector = word_vectors['front']
        elif(word == 'ç\x90ç¢ç_\x90_çâç\x81ç_rearç\x90ç¢ç_\x90_çâ\x90_ç\x9d'):
            embedding_vector = word_vectors['rear']
        elif(word == 'it´\x90´¢´_\x90_´ç´_\x90_´¢s'):
            embedding_vector = word_vectors['it']
        elif(word == '«\x90«¢«_\x90_«\x82«\x81«_front«\x90«¢«_\x90_«\x82\x90_«\x9d'):
            embedding_vector = word_vectors['front']
        elif(word == '«\x90«¢«_\x90_«\x82«\x81«_rear«\x90«¢«_\x90_«\x82\x90_«\x9d'):
            embedding_vector = word_vectors['rear']
        elif(word == 'caller«\x90«¢«_\x90_«\x82«_\x90_«¢s'):
            embedding_vector = word_vectors['caller']
        elif(word == 'array´\x90\x90_\x90_´¢´\x90´¢\x90_´ç´\x90´¢\x90_´¢s'):
            embedding_vector = word_vectors['array']
        elif(word == '´\x90´¢´_\x90_´ç´\x81´_front´\x90´¢´_\x90_´ç\x90_´\x9d'):
            embedding_vector = np.zeros(300)
        elif(word == '´\x90´¢´_\x90_´ç´\x81´_rear´\x90´¢´_\x90_´ç\x90_´\x9d'):
            embedding_vector = np.zeros(300)
        elif(word == 'caller´\x90´¢´_\x90_´ç´_\x90_´¢s'):
            embedding_vector = word_vectors['caller']
        elif(word == '´\x90´¢´_\x90_´ç´_´¾´\x9d'):
            embedding_vector = np.zeros(300)
        elif(word == '«\x90«¢«_\x90_«\x82«_«_«\x9d'):
            embedding_vector = np.zeros(300)
        elif(word == 'object«\x90«¢«_\x90_«\x82«_\x90_«¢s'):
            embedding_vector = word_vectors['object']
        elif(word == 'definition«\x90«¢«_\x90_«\x82«_\x90_«¢s'):
            embedding_vector = word_vectors['definition']
        elif(word == '«\x90«¢«_\x90_«\x82«\x8f«_int'):
            embedding_vector = np.zeros(300)
        elif(word == '«\x90«¢«_\x90_«\x82«_\x90_«¢'):
            embedding_vector = np.zeros(300)
        elif(word == 'object´\x90´¢´_\x90_´ç´_\x90_´¢s'):
            embedding_vector = word_vectors['object']
        elif(word == 'clear?'):
            embedding_vector = word_vectors['clear']
        elif(word == 'doesn\x82\x90\x82¢\x82_\x90_\x82\x89\x82_\x90_\x82¢t'):
            embedding_vector = (word_vectors['does'] + word_vectors['not'])/2
        elif(word == 'definition´\x90´¢´_\x90_´ç´_\x90_´¢s'):
            embedding_vector =  word_vectors['definition']
        elif(word == '´\x90´¢´_\x90_´ç´\x8f´_int'):
            embedding_vector = np.zeros(300)
        elif(word == '´\x90´¢´_\x90_´ç´_\x90_´¢'):
            embedding_vector = np.zeros(300)
        elif(word == 'it«\x90«¢«_\x90_«\x82«_\x90_«¢s'):
            embedding_vector = word_vectors['it']
        elif(word == 'array«\x90\x90_\x90_«¢«\x90«¢\x90_«\x82«\x90«¢\x90_«¢s'):
            embedding_vector = word_vectors['array']
        elif(word == 'statement?'):
            embedding_vector = word_vectors['statement']
        elif(word == '\x82\x90\x82¢\x82_\x90_\x82\x89\x82\x81\x82_root\x82\x90\x82¢\x82_\x90_\x82\x89\x90_\x82\x9d'):
            embedding_vector = np.zeros(300)
        elif(word == 'arrayì¢å\x90å\x90_å\x90_ì¢å¢ì¢å\x90ì¢å¢å\x90_ì¢ì_ì¢å\x90ì¢å¢å\x90_ì¢å¢s'):
            embedding_vector = word_vectors['array']
        elif(word == 'ì¢å\x90ì¢å¢ì¢_å\x90_ì¢ì_ì¢å\x8fì¢_int'):
            embedding_vector = np.zeros(300)
        elif(word == 'ì¢å\x90ì¢å¢ì¢_å\x90_ì¢ì_ì¢_å\x90_ì¢å¢'):
            embedding_vector == np.zeros(300)
        elif(word == '7*5'):
            embedding_vector = (word_vectors['7'] + word_vectors['5'])/2
        elif(word == 'â\x90â¢â_\x90_âäâ\x81â_doâ\x90â¢â_\x90_âä\x90_â\x9d'):
            embedding_vector = np.zeros(300)
        elif(word == 'ì¢å\x90å\x90_å\x90_ì¢å¢ì¢å\x90ì¢å¢å\x90_ì¢ì_ì¢å\x90å\x90_points'):
            embedding_vector = word_vectors['points']
        elif('toì¢å\x90å\x90_å\x90_ì¢å¢ì¢å\x90ì¢å¢å\x90_ì¢ì_ì¢å\x90å\x90_'):
            embedding_vector = word_vectors['to']
        elif(word == 'â\x90â¢â_\x90_âäâ\x81â_do'):
            embedding_vector = np.zeros(300)
        elif(word == '*value'):
            embedding_vector = word_vectors['value']
        elif(word == '\x82\x90\x90_\x90_\x82¢\x82\x90\x82¢\x90_\x82\x89\x82\x90\x90_points'):
            embedding_vector = word_vectors['points']
        elif(word == 'to\x82\x90\x90_\x90_\x82¢\x82\x90\x82¢\x90_\x82\x89\x82\x90\x90_'):
            embedding_vector = word_vectors['to']
        elif(word == 'arrayâ\x90\x90_\x90_â¢â\x90â¢\x90_âäâ\x90â¢\x90_â¢s'):
            embedding_vector = word_vectors['array']
        elif(word == 'whileâ\x90â¢â_\x90_âä\x90_â\x9d'):
            embedding_vector = word_vectors['while']
        elif(word == 'doesnâ\x90â¢â_\x90_âäâ_\x90_â¢t'):
            embedding_vector = word_vectors['does'] + word_vectors['not']
        elif(word == 'â\x90â¢â_\x90_âäâ\x81â_rootâ\x90â¢â_\x90_âä\x90_â\x9d'):
            embedding_vector = word_vectors['root']
        elif(word == 'â\x90\x90_\x90_â¢â\x90â¢\x90_âäâ\x90\x90_points'):
            embedding_vector = word_vectors['point']
        elif(word == 'toâ\x90\x90_\x90_â¢â\x90â¢\x90_âäâ\x90\x90_'):
            embedding_vector = word_vectors['to']
        elif(word == '*5'):
            embedding_vector = word_vectors[5]
        elif(word == 'â\x90â¢â_\x90_âäâ_â_â\x9d'):
            embedding_vector = np.zeros(300)
        elif(word == 'objectâ\x90â¢â_\x90_âäâ_\x90_â¢s'):
            embedding_vector = word_vectors['object']
        elif(word == 'definitionâ\x90â¢â_\x90_âäâ_\x90_â¢s'):
            embedding_vector = word_vectors['definition']
        elif(word == 'â\x90â¢â_\x90_âäâ\x8fâ_int'):
            embedding_vector = np.zeros(300)
        elif(word == 'â\x90â¢â_\x90_âäâ_\x90_â¢'):
            embedding_vector = np.zeros(300)
        else:
            embedding_vector = word_vectors[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
    
#creating word vector with GloVe model
def word_embed_meta_data(documents, embedding_dim):
    vocabulary_size = 20000
    tokenizer = Tokenizer(num_words= vocabulary_size, filters='')
    tokenizer.fit_on_texts(documents)
    # gloveFile = "/home/mvanessa/pastprojects/glove.6B.300d.txt"
    gloveFile = "/Users/michellevanessa/Desktop/automatic-text-scoring-master/glove.6B.300d.txt"
    model = train_word2vec(gloveFile)
    
    ## Handling the exception cases where the word in dataset is not present in the GloVe embeddings vocabulary
    model['dereferencing'] = (model['not'] + model['reference'])/2
    k = 300
    x = model['constant']
    x -= x.dot(k) * k       # make it orthogonal to k
    x /= np.linalg.norm(x)  # normalize it
    model['nonconstant'] = x
    model['enqueue'] = (model['in'] + model['queue'])/2
    model['dequeue'] = (model['out'] + model['queue'])/2
    model['initialized'] = model['initialize']
    model['intialized'] = model['initialize']
    model['5657'], model['88123'] = model['124'], model['124']
    model['precheck'] = (model['check'] + model['before'])/2
    model['typesafe'] = (model['type'] + model['safe'])/2
    model['postcheck'] = (model['check'] + model['after'])/2
    model['unmodifiable'] = (model['not'] + model['modifiable'])/2
    model['nonstatic'] = (model['not'] + model['static'])/2
    model[' '] = np.zeros(300)
    model['aptr'] = (model['a'] + model['ptr'])/2
    model['stackc'] = (model['stack'] + model['c'])/2
    model['accessmodifier'] = (model['access'] + model['modifier'])/2
    model['sublists'] = (model['sub'] + model['lists'])/2
    model['pointeroffset'] = (model['pointer'] + model['offset'])/2
    model['isemptry'] = (model['is'] + model['empty'])/2
    model['whilefor'] = (model['while'] + model['for'])/2
    model['0x000000'] = model['0000']
    model['prototyp'] = model['prototype']
    model['stringtermination'] = (model['string'] + model['termination'])/2
    model['doublylinked'] = (model['doubly'] + model['linked'])/2
    model['stackh'] = (model['stack'] + model['h'])/2
    model['preruntime'] = (model['before'] + model['runtime'])/2
    model['reinitialize'] = (model['again'] + model['initialize'])/2
    model['multiarray'] = (model['multiple'] + model['array'])/2
    model['bidimensional'] = (model['two'] + model['dimensional'])/2
    model['dereferenced'] = model['referenced']
    model['recieves'] = model['receives']
    model['arraysize'] = (model['array'] + model['size'])/2
    model['arraybased'] = (model['array'] + model['based'])/2
    model['paramater'] = model['parameter']
    model['refrenece'] = model['reference']
    model['bigo'] = (model['big'] + model['o'])/2
    model['pointerbased'] = (model['pointer'] + model['based'])/2
    model['dowhile'] = (model['do'] + model['while'])/2
    model['divideandconquer'] = (model['divide'] + model['and'] + model['conquer'])/3
    model['xptr'] = (model['x'] + model['ptr'])/2
    model['repetiive'] = model['repetitive']
    model['toget'] = (model['to'] + model['get'])/2
    model['recurisve'] = model['recursive']
    model['guaranteeed'] = model['guaranteed']
    model['cotained'] = model['contained']
    model['palces'] = model['places']
    model['dereference'] = model['referenced']
    model['dereferencing'] = model['referenced']
    model['linkbased'] = (model['link'] + model['based'])/2
    model['dereferences'] = model['referenced']
    model['recuruivly'] = model['recursively']
    model['getfront'] = (model['get'] + model['front'])/2
    model['functgion'] = model['function']
    model['struct'] = model['structure']
    model['dequeueing'] = model['dequeue']
    model['bestcase'] = (model['best'] + model['case'])/2
    model['doese'] = model['does']
    model['locala'] = model['local']
    model['progject'] = model['project']
    model['devilopment'] = model['development']
    model['heshe'] = model['hash']
    model['defination'] = model['definition']
    model['listarray'] = (model['list'] + model['array'])/2
    model['semicoln'] = model['semicolon']
    model['bublic'] = model['public']
    model['objectname'] = (model['object'] + model['name'])/2
    model['objectnamefunciton'] = (model['object'] + model['name'] + model['function'])/3
    model['differance'] = model['difference']
    model['disadvanate'] = model['disadvantage']
    model['2n1'] = (model['2'] + model['n'] + model['1'])/3
    model['beging'] = model['begin']
    model['bptr'] = (model['b'] + model['ptr'])/2
    model['fuction'] = model['function']
    model['paramaters'] = model['parameters']
    model['classname'] = (model['class'] + model['name'])/2
    model['2log'] = (model['2'] + model['log'])/2
    model['singlylinked'] = (model['singly'] + model['linked'])/2
    model['displaymessge'] = (model['display'] + model['message'])/2
    model['graduatlly'] = model['gradually']
    model['tobedeleted'] = (model['to'] + model['be'] + model['deleted'])/3
    model['accessspecifiers'] = (model['access'] + model['specifiers'])/2
    model['isempty'] = (model['is'] + model['empty'])/2
    model['miminum'] = model['minimum']
    model['subarray'] = (model['sub'] + model['array'])/2
    model['easly'] = model['easily']
    model['phasetesting'] = (model['phase'] + model['testing'])/2
    model['memberfunction'] = (model['member'] + model['function'])/2
    model['algorithyms'] = model['algorithms']
    model['backinto'] = (model['back'] + model['into'])/2
    model['insertiondeletion'] = (model['insertion'] + model['deletion'])/2
    model['selfreferential'] = (model['self'] + model['referential'])/2
    model['longes'] = model['longest']
    model['executiontime'] = (model['execution'] + model['time'])/2
    model['objectoriented'] = (model['object'] + model['oriented'])/2
    model['paremeters'] = model['parameters']
    model['attrubutes'] = model['attributes']
    model['bpointer'] = (model['b'] + model['pointer'])/2
    model['priniple'] = model['principle']
    model['inceasing'] = model['increasing']
    model['passedreturned'] = (model['passed'] + model['returned'])/2
    model['sepperate'] = model['separate']
    model['accessspecifications'] = (model['access'] + model['specification'])/2
    model['recurse'] = model['recursive']
    model['ntimes'] = (model['n'] + model['times'])/2
    model['excute'] = model['execute']
    model['rightptr'] = (model['right'] + model['pointer'])/2
    model['opperation'] = model['operation']
    model['lastin'] = (model['last'] + model['in'])/2
    model['firstout'] = (model['first'] + model['out'])/2
    model['conditionsinputs'] = (model['condition'] + model['input'])/2
    model['leftptr'] = (model['left'] + model['pointer'])/2
    model['enqueued'] = model['enqueue']
    model['wmultiple'] = model['multiple']
    model['queuefront'] = (model['queue'] + model['front'])/2
    model['structs'] = model['structure']
    model['recursionrepeated'] = (model['recursion'] + model['repeated'])/2
    model['iterationlooptermination'] = (model['iteration'] + model['loop'] + model['termination'])/3
    k = 300
    x = model['allocate']
    x -= x.dot(k) * k       # make it orthogonal to k
    x /= np.linalg.norm(x)  # normalize it
    model['deallocate'] = x
    model['recursionbase'] = (model['recursion'] + model['base'])/2
    model['iterationmodifies'] = (model['iteration'] + model['modify'])/2
    model['looptermination'] = (model['loop'] + model['termination'])/2
    model['recursionproduces'] = (model['recursion'] + model['produces'])/2
    model['arrayname'] = (model['array'] + model['name'])/2
    model['iterationif'] = model['iteration']
    model['iterationrepetition'] = (model['iteration'] + model['repitition'])/2
    model['recursionselection'] = (model['recursion'] + model['selection'])/2
    model['iterationexplicitly'] = (model['iteration'] + model['explicitly'])/2
    model['whenevery'] = (model['when'] + model['every'])/2
    model['specifiy'] = model['specify']
    model['entitiy'] = model['entity']
    model['limitationdeclaration'] = (model['limitation'] + model['declaration'])/2
    model['coquer'] = model['conquer']
    model['evaulate'] = model['evaluate']
    model['deallocateddeleted'] = (model['deallocate'] + model['deleted'])/2
    model['repetative'] = model['repetitive']
    model['opertation'] = model['operation']
    model['programmar'] = model['programmer']
    model['amout'] = model['amount']
    model['opertator'] = model['operator']
    model['comonly'] = model['commonly']
    model['opperations'] = model['operations']
    model['inplace'] = (model['in'] + model['place'])/2
    model['invovles'] = model['involves']
    model['explicityly'] = model['explicitly']
    model['earch'] = model['each']
    model['inlineexpands'] = (model['inline'] + model['expands'])/2
    model['ooadesign'] = (model['object'] + model['oriented'] + model['design'])/3
    model['employee1'] = (model['employee'] + model['1'])/2
    model['dynamicly'] = model['dynamically']
    model['recursuivly'] = model['recursively']
    model['togather'] = model['together']
    model['listbased'] = (model['list'] + model['based'])/2
    model['valueword'] = (model['value'] + model['word'])/2
    model['intialize'] = model['initialize']
    model['linkedbase'] = (model['link'] + model['based'])/2
    model['flaxible'] = model['flexible']
    model['intializes'] = model['initialize']
    model['meomrry'] = model['memory']
    model['sizeof'] =(model['size'] + model['of'])/2
    model['concatenate'] = model['add']
    model['ausednumber'] = (model['used']+model['number'])/2
    model['resizeable'] = model['resize']
    model['dequeued'] = model['dequeue']
    model['ispalindrome'] = model['palindrome']
    model['ponter'] = model['pointer']
    model['excutes'] = model['execute']
    model['ofdata'] = model['data']
    model['traversals'] = model['traverse']
    model['nlog'] = (model['n'] + model['log'])/2
    model['entitity'] = model['entity']
    model['inputing'] = model['input']
    model['recursionif'] = model['recursion']
    model['postorder'] = (model['post'] + model['order'])/2
    model['deque'] = model['dequeue']
    model['infinitly'] = model['infinitely']
    model['repition'] = model['repitition']
    model['instantiating'] = model['instantiate']
    model['listbases'] = (model['list'] + model['based'])/2
    model['delcared'] = model['declared']
    model['consits'] = model['consists']
    model['enque'] = model['enqueue']
    model['inorder'] = model['order']
    model['passbyvalue'] = (model['pass'] + model['value'])/2
    model['specifacations'] = model['specifications']
    model['multideminsional'] = model['multidimensional']
    model['loopcontinuation'] = (model['loop'] + model['continuation'])/2
    model['fixedsized'] = (model['fixed'] + model['sized'])/2
    model['filescope'] = (model['file'] + model['scope'])/2
    model['myclass'] = model['class']
    model['insertingdeleting'] = (model['insert']+model['delete'])/2
    model['growshrink'] = (model['grow'] + model['shrink'])/2
    model['implementatoin'] = model['implementation']
    model['entinty'] = model['entity']
    model['elementindex'] = (model['element'] + model['index'])/2
    model['passbyreference'] = (model['pass'] + model['reference'])/2
    model['\x92'] = model['92']
    model['sybol'] = model['symbol']
    model['dolook'] = model['look']
    model['valuenumber'] = (model['value'] + model['number'])/2
    model['program;'] = model['program']
    model['lastinfirstout'] = (model['last'] + model['first'])/2
    model['firstinfirstout'] = (model['first'] + model['first'])/2
    model['insertend'] = (model['insert'] + model['end'])/2
    model['deleteend'] = (model['delete'] + model['end'])/2
    model['stackdisplay'] = (model['stack'] + model['display'])/2
    model['initialises'] = model['initialize']
    model['varioubles'] = model['variables']
    model['doublelinked'] = (model['double'] + model['linked'])/2
    model['prototye'] = model['prototype']
    model['wantedneeded'] = (model['wanted'] + model['needed'])/2
    model['poiner'] = model['pointer']
    model['encapsulationobjects'] = (model['encapsulation'] + model['objects'])/2
    model['solvin'] = model['solving']
    model['retreive'] = model['retrieve']
    model['bigoh'] = (model['big'] + model['oh'])/2
    model['cirtain'] = model['certain']
    model['unessesary'] = model['unnecessary']
    model['localvariable'] = (model['local'] + model['variable'])/2
    model['recurses'] = model['recursive']
    model['traveral'] = model['traverse']
    model['shiftnig'] = model['shifting']
    model['scheduleing'] = model['scheduling']
    model['perfom'] = model['perform']
    model['forsight'] = model['foresight']
    model['challanges'] = model['challenges']
    model['initilizer'] = model['initialize']
    model['funtions'] = model['functions']
    model['specificly'] = model['specifically']
    model['refrence'] = model['reference']
    model['bydimensional'] = model['bidimensional']
    model['2non'] = (model['2'] + model['non'])/2
    model['strlen'] = (model['str'] + model['len'])/2
    model['ammout'] = model['amount']
    model['mergesort'] = (model['merge'] + model['sort'])/2
    model['subarrays'] = (model['sub'] + model['arrays'])/2
    model['incorerectly'] = model['incorrectly']
    model['arrarys'] = model['arrays']
    model['unqiue'] = model['unique']
    model['logarthmic'] = model['logarithmic']
    model['prenthesis'] = model['parenthesis']
    model['arrayptr'] = (model['array'] + model['ptr'])/2
    model['logn'] = (model['log'] + model['n'])/2
    model['postprogramming'] = (model['post'] + model['programming'])/2
    model['elimitates'] = model['eliminates']
    model['amonts'] = model['amounts']
    model['avariable'] = (model['a'] + model['variable'])/2
    model['isdeclared'] = (model['is'] + model['declared'])/2
    model['complier'] = model['compiler']
    model['throughought'] = model['throughout']
    model['mbyn'] = (model['m'] + model['by'] + model['n'])/3
    model['rowcolumn'] = (model['row'] + model['column'])/2
    model['firstin'] = (model['first'] + model['in'])/2
    model['modularability'] = model['modularity']
    model['parenthisis'] = model['parenthesis']
    model['initilized'] = model['initialize']
    model['reinitilized'] = (model['re'] + model['initialized'])/2
    model['initilization'] = model['initialize']
    model['oneway'] = (model['one'] + model['way'])/2
    model['infinetly'] = model['infinitely']
    model['fucntions'] = model['functions']
    model['modifie'] = model['modify']
    model['debugg'] = model['debug']
    model['comperison'] = model['comparison']
    model['arangment'] = model['arrangement']
    model['arraybases'] = (model['array'] + model['based'])/2
    model['pointerbases'] = (model['pointer'] + model['based'])/2
    model['methodsclasses'] = (model['methods'] + model['classes'])/2
    model['mygradebook'] = (model['my'] + model['grade'] + model['book'])/3
    model['msort'] = (model['merge'] + model['sort'])/2
    model['mid1'] = (model['mid'] + model['1'])/2
    model['sourcecode'] = (model['source'] + model['code'])/2
    model['explictly'] = model['explicitly']
    model['consructor'] = model['constructor']
    model['simplier'] = model['simpler']
    model['1initializing'] = (model['1'] + model['initialize'])/2
    model['2specifying'] = (model['2'] + model['specifying'])/2
    model['concatenates'] = model['add']
    model['delcaration'] = model['declaration']
    model['nonconst'] = (model['non'] + model['constant'])/2
    model['const'] = model['constant']
    model['twor'] = model['two']
    model['eaach'] = model['each']
    model['infinately'] = model['infinitely']
    model['linkedlist'] = (model['linked'] + model['list'])/2
    model['isfull'] = (model['is'] + model['full'])/2
    model['isfullq'] = (model['is'] + model['full'] + model['queue'])/3
    model['maxqueuesize1'] = (model['max'] + model['queue'] + model['size'] + model['1'])/4
    model['pushpop'] = (model['push'] + model['pop'])/2
    model['oposite'] = model['opposite']
    model['reusuablitly'] = model['reusability']
    model['iterativly'] = model['iteratively']
    model['aliasnickname'] = (model['alias'] + model['nickname'])/2
    model['programer'] = model['programmer']
    model['userdefined'] = (model['user'] + model['defined'])/2
    model['agian'] = model['again']
    model['4005'] = (model['4000'] +model['5'])
    model['3580'] = (model['3500'] + model['80'])
    model['sourted'] = model['sorted']
    model['funcion'] = model['function']
    model['gettop'] = (model['get'] + model['top'])/2
    model['compilertime'] = (model['compile'] + model['time'])/2
    model['differntiated'] = model['differentiated']
    model['enqueueing'] = model['enqueue']
    model['dequeueing'] = model['dequeue']
    model['feused'] = model['reused']
    model['splitproblems'] = (model['split'] + model['problems'])/2
    model['bidimensionaltwodimensional'] = (model['bidimensional'] + model['two'] + model['dimensional'])/3
    model['12345678910'] = (model['1'] + model['2'] + model['3'] + model['4'] + model['5'] + model['6'] + model['7'] + model['8'] + model['9'] + model['10'])/10
    model['accesed'] = model['accessed']
    model['comon'] = model['common']
    model['arrary'] = model['array']
    model['acordingly'] = model['accordingly']
    model['listbase'] = (model['list'] + model['based'])/2
    model['addedsubtracted'] = (model['added'] + model['subtracted'])/2
    model['prespecified'] = (model['pre'] + model['specified'])/2
    model['stringline'] = (model['string'] + model['line'])/2
    model['sepereate'] = model['separate']
    model['smallers'] = model['smaller']
    model['funciton'] = model['function']
    model['8123'] = model['8000'] + model['123']
    model['maxqueuesize'] = (model['max'] + model['queue'] + model['size'])/3
    model['dimensionaltwo'] = (model['dimensional'] + model['two'])/2
    model['deallocated'] = model['deallocate']
    model['ooa'] = (model['object'] + model['oriented'])/2
    model['coninuously'] = model['continuously']
    model['attatched'] = model['attached']
    model['uninitialize'] = (model['not'] + model['initialize'])/2
    model['flexablity'] = model['flexibility']
    model['curptr'] = (model['current'] + model['pointer'])/2
    model['itteration'] = model['iteration']
    model['conqure'] = model['conquer']
    model['ptrtoarray'] = (model['pointer'] + model['to'] + model['array'])/3
    model['nthentity'] = (model['n'] + model['entity'])/2
    model['quese'] = model['queue']
    model['cout'] = (model['c'] + model['out'])/2
    model['indiviually'] = model['individually']
    model['furtherest'] = model['furthest']
    model['arraylist'] = (model['array'] + model['list'])/2
    model['atributes'] = model['attributes']
    model['halfs'] = model['halves']
    model['leastanswer'] = (model['least'] + model['answer'])
    model['ecetera'] = model['etc']
    model['arraycontinually'] = (model['array'] + model['continually'])/2
    model['scalablelist'] = (model['scalable'] + model['list'])/2
    model['definesdata'] = (model['defines'] + model['data'])
    model['problemiteration'] = (model['problem'] + model['iteration'])/2
    model['stackpush'] = (model['stack'] + model['push'])/2
    model['stackisempty'] = (model['stack'] + model['is'] + model['empty'])/3
    model['createstack'] = (model['create'] + model['stack'])/2
    model['easer'] = model['easier']
    model['vistit'] = model['visit']
    model['stackpop'] = (model['stack'] + model['pop'])/2
    model['ptnode'] = (model['pointer'] + model['node'])/2
    model['containersdivides'] = (model['container'] + model['divides'])/2
    word_vector = {}
    word_vector['while'] = model['while']
    word_vector['does'] = model['does']
    word_vector['not'] = model['not']
    word_vector['3.5'] = model['3.5']
    word_vector['*bptr'] = (model['b'] + model['ptr'])/2
    word_vector['?'] = model['?']
    word_vector['5'] = model['5']
    word_vector[5] = model['5']
    word_vector['to'] = model['to']
    word_vector['it'] = model['it']

    for i in range(len(documents)):
        tokens = word_tokenize(documents[i])
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        #remove remaining tokens that are not alphanumeric
        words = [word for word in stripped if word.isalnum()]
        for word in words:
            word_vector[word] = model[word]
    embedding_matrix = create_embedding_matrix(tokenizer, word_vector, embedding_dim)
    del word_vector
    gc.collect()
    return tokenizer, embedding_matrix
 
#creating training dataset...   
def create_train_dev_set(tokenizer, answers_pair, feat, ans_grade, max_sequence_length, validation_split_ratio):
        
    answers1 = [x[0] for x in answers_pair]
    answers2 = [x[1] for x in answers_pair]
    train_answers_1 = tokenizer.texts_to_sequences(answers1)
    train_answers_2 = tokenizer.texts_to_sequences(answers2)
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
             for x1, x2 in zip(train_answers_1, train_answers_2)]
    len_answer = feat['Length Answer'].to_numpy()
    len_ref_answer = feat['Length Ref Answer'].to_numpy()
    len_ref_by_len_ans = feat['Len Ref By Ans'].to_numpy()
    words_ans = feat['Words Answer'].to_numpy()
    uniq_words_ans = feat['Unique Words Answer'].to_numpy()
    
    train_padded_data_1 = pad_sequences(train_answers_1, maxlen=max_sequence_length, padding='post')
    train_padded_data_2 = pad_sequences(train_answers_2, maxlen=max_sequence_length, padding='post')
    train_scores = np.array(ans_grade)
    leaks = np.array(leaks)
    
    shuffle_indices = np.random.permutation(np.arange(len(train_scores)))
    train_data_1_shuffled = train_padded_data_1[shuffle_indices]
    train_data_2_shuffled = train_padded_data_2[shuffle_indices]
    train_scores_shuffled = train_scores[shuffle_indices]
    leaks_shuffled = leaks[shuffle_indices]
    len_answer_shuffled = len_answer[shuffle_indices]
    len_ref_answer_shuffled = len_ref_answer[shuffle_indices]
    len_ref_by_len_ans_shuffled = len_ref_by_len_ans[shuffle_indices]
    words_ans_shuffled = words_ans[shuffle_indices]
    uniq_words_ans_shuffled = uniq_words_ans[shuffle_indices]
    
    dev_idx = max(1, int(len(train_scores_shuffled) * validation_split_ratio))
    print(train_data_2_shuffled, train_data_1_shuffled)
    del train_padded_data_1
    del train_padded_data_2
    gc.collect()
    
    train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
    train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
    scores_train, scores_val = train_scores_shuffled[:-dev_idx], train_scores_shuffled[-dev_idx:]
    leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]
    len_answer_train, len_answer_val = len_answer_shuffled[:-dev_idx], len_answer_shuffled[-dev_idx:]
    len_ref_answer_train, len_ref_answer_val = len_ref_answer_shuffled[:-dev_idx], len_ref_answer_shuffled[-dev_idx:]
    len_ref_by_len_ans_train, len_ref_by_len_ans_val = len_ref_by_len_ans_shuffled[:-dev_idx], len_ref_by_len_ans_shuffled[-dev_idx:]
    words_ans_train, words_ans_val = words_ans_shuffled[:-dev_idx], words_ans_shuffled[-dev_idx:]
    uniq_words_ans_train, uniq_words_ans_val = uniq_words_ans_shuffled[:-dev_idx], uniq_words_ans_shuffled[-dev_idx:]
    
    feat_train_dict = {}
    feat_train_dict['Len Answer'] = len_answer_train
    feat_train_dict['Len Ref Answer'] = len_ref_answer_train
    feat_train_dict['Len Ref By Ans'] = len_ref_by_len_ans_train
    feat_train_dict['Words Ans'] = words_ans_train
    feat_train_dict['Uniq Words Ans'] = uniq_words_ans_train
    feat_train = pd.DataFrame(feat_train_dict, columns=['Len Answer', 'Len Ref Answer', 'Len Ref By Ans', 'Words Ans', 'Uniq Words Ans'])

    feat_val_dict = {}
    feat_val_dict['Len Answer'] = len_answer_val
    feat_val_dict['Len Ref Answer'] = len_ref_answer_val
    feat_val_dict['Len Ref By Ans'] = len_ref_by_len_ans_val
    feat_val_dict['Words Ans'] = words_ans_val
    feat_val_dict['Uniq Words Ans'] = uniq_words_ans_val
    feat_val = pd.DataFrame(feat_val_dict, columns=['Len Answer', 'Len Ref Answer', 'Len Ref By Ans', 'Words Ans', 'Uniq Words Ans'])
    
    return train_data_1, train_data_2, scores_train, leaks_train, feat_train, val_data_1, val_data_2, scores_val, leaks_val, feat_val

## Create the test data
def create_test_data(tokenizer, test_answers_pair, feat, max_sequence_length):
   
    test_answers1 = [x[0] for x in test_answers_pair]
    test_answers2 = [x[1] for x in test_answers_pair]
    
    test_answers_1 = tokenizer.texts_to_sequences(test_answers1)
    test_answers_2 = tokenizer.texts_to_sequences(test_answers2)
    leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                  for x1, x2 in zip(test_answers_1, test_answers_2)]
    leaks_test = np.array(leaks_test)
    test_data_1 = pad_sequences(test_answers_1, maxlen=max_sequence_length, padding='post')
    test_data_2 = pad_sequences(test_answers_2, maxlen=max_sequence_length, padding='post')
   
    return test_data_1, test_data_2, feat, leaks_test
