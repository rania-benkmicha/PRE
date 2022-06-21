import rosbag, sys, csv
import time
import string
import os 
import cv2
from cv_bridge import CvBridge
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import io
from PIL import Image
import rospy






debut = time.time()

# Variables
listOfImageTopics = ['/zed_node/left/image_rect_color']#, '/zed_node/depth/depth_registered'] # Liste des topics avec images qu'on veut sauvegarder
                                                                                            # /!\ a mettre dand l'ordre des priorites si un topic 
                                                                                            # est plus important que l'autre. Mettre en premier le topic principal
                                                                                            #a modifier
listOfTopics = ["/imu/data"] #Autres topics qu'on veut sauvegarder
#a modifier
downsamplingRatio = 20 #Downsamplig ratio, par exemple s'il vaut 5, une donnee sur 5 sera enregistree
#a modifier
epsilon = 0.1 # Les differents topics ne publient pas en meme temps. epsilon correspond a l'ecart de temps qu'on s'autorise (seconde)
#a modifier


# Verifie les arguments du script. On peut soit appeler le script en tapant "python read_bags.py" soit "python read_bags.py nom_du_rosbag.bag"
if (len(sys.argv) > 2):
    print("invalid number of arguments:   " + str(len(sys.argv)))
    print("should be 2: 'read_bags.py' and 'bagName'")
    print("or just 1: 'read_bags.py'")
    sys.exit(1)
elif (len(sys.argv) == 2):
    listOfBagFiles = [sys.argv[1]]
    numberOfFiles = "1"
    print("reading only 1 bagfile: " + str(listOfBagFiles[0]))
elif (len(sys.argv) == 1):
    listOfBagFiles = [f for f in os.listdir(".") if f[-4:] == ".bag"]	#get list of only bag files in current dir.
    numberOfFiles = str(len(listOfBagFiles))
    print("reading all " + numberOfFiles + " bagfiles in current directory: \n")
    for f in listOfBagFiles:
        print(f)
else:
    print("bad argument(s): " + str(sys.argv))	#shouldnt really come up
    sys.exit(1)


count = 0
for bagFile in listOfBagFiles: # On fait les calculs/sauvegardes pour chaque rosbags si plusieurs rosbags
    count += 1
    print("reading file " + str(count) + " of  " + numberOfFiles + ": " + bagFile)
    #acces au rosbag
    bag = rosbag.Bag(bagFile)
    bagContents = bag.read_messages()
    bagName = bag.filename
    #cree un nouveau dossier
    directory = os.path.abspath(os.getcwd())
    results_dir = directory + "/Bags_Data_" + bagName[:-4]
    try:	#on cree le nouveau dossier suelement s'il n'existe pas deja 
        os.mkdir(results_dir)
        print(results_dir + " folder created")
    except:
        pass


    numberOfImages = []

    # Calcul du nombres de donnees a enregistrer en se basant sur le nombre d'images dans le flux video. 
    # Le flux video est le facteur limitant en terme de nombre de donnees. 






    for i in range(len(listOfImageTopics)): # pour chaque flux d'image : 

        ## Calcul de la nouvelle frequence d'echantillonage en fonction du downsampling ratio
        max_frequency = bag.get_type_and_topic_info()[1][str(listOfImageTopics[i])][3]
        numberOfImages.append(bag.get_type_and_topic_info()[1][str(listOfImageTopics[i])][1])
        print("\nThere are " + str(numberOfImages[i]) + " images in the " + listOfImageTopics[i] + " topic.")
        print(listOfImageTopics[i] + " frequency is %.2f Hz. Downsampling ratio is " % max_frequency + str(downsamplingRatio) + ".")
        new_frequency = max_frequency/downsamplingRatio
        # Demande confirmation pour continuer en fonction de la nouvelle frequence moyenne des donnees
        # Pour arreter, ecrire "No" dans le terminal. Sinon, pour continuer, n'importe quelle touche fonctionne en plus de "Yes"
        print("New mean frequency will be %.2f Hz. Continue ?" % new_frequency)
        answer = input("Yes/No\n")
        if answer == "No":
            exit()


    distance=int(input('ecrivez la distance entre capteur et pont capturé'))
    rayon_de_roue=int(input('donnez le rayon de roue'))
    #delta_t=distance/vitesse_angualaire*rayon_de_roue
    
    
    listTimestampImages = [] # timestamp = nom de l'indicateur de temps pour ros. Utile car ne depend pas du topic, reste le meme d'un topic a l'autre
    # on garde les timestamp pour ensuite pouvoir recuperer les donnees qui correspondent aux instants ou les images ont ete prises. 
    listnum=[]
    # pour chaque flux d'image : 
    for i in range(len(listOfImageTopics)):
        print(listOfImageTopics[i])

        bridge = CvBridge()
        topicName = listOfImageTopics[i].replace('/', '_') 
        topicDir = results_dir + "/" + topicName
        count_i = 0

        # creation d'un dossier s'il n'existe pas deja 
        try : 
            os.mkdir(topicDir)
        except: 
            pass

        csvname = topicDir + "/" + topicName + ".csv" # creation d'un csv avec les timestamp
        with open(csvname, 'w') as csvfile: 
            filewriter = csv.writer(csvfile, delimiter = ',')

            if i == 0: # On enregistre les donnees de maniere naive pour le 1er topic, le topic "principal"
                num=1
                for topic, msg, t in bag.read_messages(topics=listOfImageTopics[i]):
                    if count_i == 0:
                        filewriter.writerow(["TimeStamp"]) # ecriture 1ere ligne du csv
                        t0 = t

                    if count_i%downsamplingRatio == 0: # on sauvegarde uniquement une donnee sur "downsamplingRatio"
                        cv_img = bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
                        im = Image.fromarray(cv_img)
                        im = im.convert("RGB")
                        
                        im.save(topicDir + "/" + str(num) + ".png", "PNG")
                        #im.save(topicDir + "/" + str(t) + ".png", "PNG")
                        
                        listTimestampImages.append(rospy.Time.to_sec(t)) # on remplit la liste avec les instants qu'on veut sauvegarder
                        listnum.append(str(num))
                        filewriter.writerow([str(t)])
                    count_i+=1
                    num=num+1

            else: # Pour les autres topic avec images on enregistre les donnees qui correspondent aux images "principales" s'il y en a 
                        # avec une tolerance de epsilon seconde
                
                last_t = t0
                for t_image in listTimestampImages:
                    for topic, msg, t in bag.read_messages(topics=listOfImageTopics[i], start_time=last_t):
                        if rospy.Time.to_sec(t) < t_image + epsilon:   # on sauvegarde uniquement si l'instant est assez proche 
                                                                                                            # des images deja sauvegardees
                            cv_img = bridge.imgmsg_to_cv2(msg,desired_encoding="passthrough")
                            im = Image.fromarray(cv_img)
                            im = im.convert("RGB")
                            #pour controler le nom de l'image
                            
                            im.save(topicDir + "/" + str(t) + ".png", "PNG")
                            
                            
                            filewriter.writerow([str(t)])

                            last_t = t
                     
                            break #si une image correspond a un timestamp on arrête de parcourir le rosbag et on passe aux images du topic principal suivantes. 
                
    #print(len(listnum))
    for topicName in listOfTopics:
        print(topicName)
        # Create a new CSV file for each topic
        filename = results_dir + '/' + str.replace(topicName, '/', '_') + '.csv'
        with open(filename, 'w+') as csvfile:
            filewriter = csv.writer(csvfile, delimiter = ',')
            firstIteration = True	#allows header row

            last_t = t0
            for t_image in listTimestampImages: # on compare chaque instant du topic avec les instants "t_image" ou nous avons sauvegarde une image 

                for subtopic, msg, t in bag.read_messages(topicName, start_time=last_t):	# for each instant in time that has data for topicName
                    #parse data from this instant, which is of the form of multiple lines of "Name: value\n"
                    #	- put it in the form of a list of 2-element lists


   #****pour compter le retatd des données % aux images
   
                    msgString1 = str(msg)
                    msgList1 = str.split(msgString1, '\n')
                    pos=msgList1.index('angular_velocity: ')
                    vitesse_angulaire=msgList1[pos+2]
                    msgList1=msgList1[pos+2]
                    splitPair1 = str.split(msgList1, ':')
                    print(splitPair1)
                    delta_t=distance/float(splitPair1[1])*rayon_de_roue
                    print(delta_t)

                    if rospy.Time.to_sec(t) < t_image + epsilon  and rospy.Time.to_sec(t) > t_image - epsilon: # on sauvegarde uniquement si l'instant est assez proche 
                                                                                                    # des images deja sauvegardees

                        # enlever ce "if" et le "break" ligne 186 si on veut les donnees sur la mission entiere  
                        #print (subtopic)
                        last_t = t
                        #je veux les msgs qui concernent  angular velocity
                        msgString = str(msg)
                        #print(msg)
                        msgList = str.split(msgString, '\n')
                        #print(msgList)
                        instantaneousListOfData = []
                        pos=msgList.index('angular_velocity: ')
                        pos1=msgList.index('linear_acceleration: ')
                         #je veux les msgs qui concernent  angular velocity
                        msgList=msgList[pos:pos+4]+msgList[pos1:pos1+2]
                        
                        
                        #print(msgList)
                        for nameValuePair in msgList:
                            splitPair = str.split(nameValuePair, ':')
                            for i in range(len(splitPair)):	#should be 0 to 1
                                splitPair[i] = str.strip(splitPair[i])
                            instantaneousListOfData.append(splitPair)
                        #print(instantaneousListOfData)
                        
                        
                        #write the first row from the first element of each pair
                        if firstIteration:	# header
                            headers = ["rosbagTimestamp"]	#first column header
                            for pair in instantaneousListOfData:
                                headers.append(pair[0])
                            filewriter.writerow(headers)
                            firstIteration = False
                        # write the value from each pair to the file
                        #values = [str(t)]
                        values = [listnum[listTimestampImages.index(t_image)]+'.png']
                        #print(values)	#first column will have rosbag timestamp
                        for pair in instantaneousListOfData:
                        
                            if len(pair) > 1 :
                                values.append(pair[1])
                        filewriter.writerow(values)

                        break #si une donnee IMU correspond a un timestamp image on passe aux donnees IMU suivantes. 
                                # on ne veut pas que pour une image il y ait plusieurs donnees imu

    bag.close()

print("\nDone reading all " + numberOfFiles + " bag files.")

fin = time.time()
print("Le traitement des donnees a dure " + str(round(fin - debut, 1)) + " s.")
