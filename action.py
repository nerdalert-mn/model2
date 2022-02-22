import os 
from datetime import datetime
import requests

import firebase_admin
from firebase_admin import credentials, firestore, storage

dir_path = os.path.dirname(os.path.realpath(__file__))

cred = credentials.Certificate(f"{dir_path}/firebase_admin.json")
firebase_admin.initialize_app(cred, {
'storageBucket': 'rtvd-test.appspot.com'
})
db = firestore.client()

def send_notification(message =  "Сэжигтэй зүйл илэрлээ"):

    for user in db.collection('users').get():
        print(user.get('fcmToken'))
        if user.get('type') == 'normal':
            print('send notif')
            resp = requests.post('https://fcm.googleapis.com/fcm/send', json={
            "to": user.get('fcmToken'),
            "notification": {
            "title": message
            },
            "android": {
                "priority": "high",
                "notification": {
                    "channel_id": "high_importance_channel"
                }
            }
        }, headers={
            'Authorization': 'key=AAAAJZZnD84:APA91bGF2CxejvYww1JD4ssMwNltCEnFu_zmWkkXEbzL6rWzH02079Si4Yp_EVhmEbP38hC5In2Ma9rj3ynqUXu_vAaM2xiXjdWfQ0grPEIaHCiK1F2ovKJSjirZUJgyFYQOAHdtHAqe '
        })
    docRef = db.collection('history').document()
    id = docRef.id
    filename = f'{id}.jpg'
    data = {
    'timestamp': datetime.now(),
    'fileName': filename
    }
    blob = storage.bucket().blob(f'detection_images/{filename}')
    blob.upload_from_filename('./detection.jpg', content_type='image/jpeg')
    blob.make_public()

    docRef.set(data)
    db.collection('new_detections').document('recent').set(data) # duplication