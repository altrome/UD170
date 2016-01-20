import numpy
import pandas
import statsmodels.api as sm

def simple_heuristic(file_path):
    '''
    The available attributes are:
    Pclass          Passenger Class
                    (1 = 1st; 2 = 2nd; 3 = 3rd)
    Name            Name
    Sex             Sex
    Age             Age
    SibSp           Number of Siblings/Spouses Aboard
    Parch           Number of Parents/Children Aboard
    Ticket          Ticket Number
    Fare            Passenger Fare
    Cabin           Cabin
    Embarked        Port of Embarkation
                    (C = Cherbourg; Q = Queenstown; S = Southampton)
    '''

    predictions = {}
    df = pandas.read_csv(file_path)
    #print df[['Sex', 'Age', 'SibSp', 'Parch', 'Pclass' ]][(df.Survived == 0) & (df.Sex == 'female')]
    #print df
    for passenger_index, passenger in df.iterrows():
        passenger_id = passenger['PassengerId']

        if ((passenger['Sex'] == 'female')
            or
            (passenger['Age'] < 13 and passenger['Pclass'] == 2 )
            or
            (passenger['Pclass'] == 1 and passenger['Fare'] > 300)
            or
            (passenger['Pclass'] == 1 and passenger['Age'] > 75 )):
            if passenger['Survived'] == 1:
                predictions[passenger_id] = 1
            else:
                predictions[passenger_id] = 0
        else :
            if passenger['Survived'] == 0:
                predictions[passenger_id] = 1
            else:
                predictions[passenger_id] = 0   
    print 'Your heuristic is '+str(round(((sum(predictions.values())/float(len(predictions)))*100),2))+'% accurate.'
    #return predictions
simple_heuristic('train.csv')
