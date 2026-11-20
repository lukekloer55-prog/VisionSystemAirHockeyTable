"""Stable champion identities and score tables, separate from policy filenames."""
NAMES=['Julius Caesar','Athena','Leonidas','Joan of Arc','Boudicca','Hannibal','Mulan','Saladin','Artemisia','Scipio','Tomoe Gozen','Alexander']
def hero(number):
    return NAMES[number%len(NAMES)]+(' '+str(number//len(NAMES)+1) if number>=len(NAMES) else '')
def rankings(summary,names):
    return [{'rank':i+1,'model':key,'name':names.get(key,key),'points':points,'historical_points':summary['historical_points'].get(key,0)} for i,(key,points) in enumerate(sorted(summary['points'].items(),key=lambda x:(-x[1],x[0])))]
