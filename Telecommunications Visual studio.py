import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.title('Telicommunications churn')
st.image('https://editor.analyticsvidhya.com/uploads/94357telecom%20churn.png')
st.sidebar.image('data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAoHCBUSFRUWFhYZFBgWHBUZGRoYGRoaHBwVHBoaGhkWHBwcIS4mHB4sHxoZJjgmLDExNTVDHCQ7Tkg0Py41NTEBDAwMEA8QHhISHzQsJSs4NDQ0PTE/MTQ2NDQxNDE2MTU9NDE0NDQxNDQ0NjQ0ND80ODg0NDQxNDQ2NjY0NDQxPf/AABEIAHIBuAMBIgACEQEDEQH/xAAbAAEAAwEBAQEAAAAAAAAAAAAAAgMEAQUGB//EAD4QAAICAQMCAwUGAwYFBQAAAAECABEhAxIxQVEEImEFMnGBkRNCUmJyoYKSsRQjssHR8AYzU+HxQ3OTotL/xAAZAQEBAQEBAQAAAAAAAAAAAAAAAQIDBAX/xAAiEQEBAAICAgIDAQEAAAAAAAAAAQIRAzESIQQTQVFhgQX/2gAMAwEAAhEDEQA/APzyIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIlnhdLe6Jdb2Vbq63MBddeZ6/tX2Imijursdm21ZdrAswWj07nBJx8a3jx5ZY3KdTtzy5cccpjlfd6eJERMOhERAROEzogIhlI5BGAcisEAg/Aggj4wqkkAAkkgAAWSTgAAcmAiaG00Q01u3UIwCg9t9Hcfhj1MhvQ8oV/Sxx8mu/hYgVRLdXSoBgdynAaqz+Fh91vTPoTKoCIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICWaWnuvOemL789hjn1EriAiaPtRqe+abpqUSfg4HvD8w8w/NQE6ng3ayNu1TRYugS+aDE0T8I1sZoktXTZCVYUR0P7H1EjAThYTs9P2H7QfT1EUarIls237RtNGcKSgeiBsLhFYngEwM2j7M19RQ6aOo6nIZUchh3Wh5h8JlIokHBBIIPIIwQexm/24/2moruVfVZE+2KlWX7QFlFFPLewadhcBt3wk01T4lGVzu1UVn03OWdEG59Jzy1IGZScjaV4K7Q82IiB1HKkMOVII+INib/E+131FKOFKtsugQ3kPlIN81jt88zz4msc8sZZL6rGXHjllMrPcW6ujQDA7kOAwxn8LD7renWjRIzKpPS1ChsdcEHII7EdR/vmWHSD5TnqnJHqv4l9OR6gFpltRL/DeG37yWVFQAszbiBbBQAEUkkk9B0J6SiaPB+ICFg4LI67HANNt3BgynjcrKrC8GiDgmBo8Leg7qz7GZKTUWyFLbHTUUqNwVk8u4DcBqE1YqafGeFUrp6urqINxdNT7Nld9Rk2nemy1VirqpL1ldxDFqMdbw+m+jpt9utabNpk7NTeEb+801K1t3bvt8b9uFG6YPGeID7VQFUQbUBNtRJZnY8bmJJNYGBwogQ8TrnUdnIA3G6HCjhUHoAAB8BJeHbaHbrQVT2ZuW/lVx/ED0lEu0U3K46gbwO+33h/KWb+EwO+B8N9q6JYXcas8AUSTQ544mz2h7HbRTfvVxajy82wvP4eCM185n0HOg6vw6m9vBXBHmP3Wzxz3rg6vG+1v7QNjKE3Nv3WSN+2jYrgkDjizzxO+H1+F8u/w4Z/b9k8evyweGNkp0cVX5vuH+avkWHWUy/T0CC+8EBFJb41SCxdgsVyOhvjMonB3IiICIiAiJLT0yxpQSecdu/wgRiS1EKmmBB9e3f1EaaFjQ+JJ4AHJPpAjEvGkjWFYkgE0VABrJohjmrOefjiURoIiAfn6f5YgIktVwWJChQSSFBJodrPMjAREQEREBERARBFYOD6xATqVYu661g/XpORA0IiOdoDIxwtsGBPRTgVff8A8iGjp3uLEqq5bGeaCgdyf6E9JVLdXxDOKNZNkgAFmzTMRycn6nvLtR9hFruUjoxDWPQgCvhUqiJEIiICIgC/W/69oCa20y6IAQCgbysQthjuDruoEEUP4ROFF0r3U+pxtoFE/X0d/wAvA63lRmdyxLMSxOSSbJPqZZVlafGAUigh9iUzLlRbkhQeoG4C+JlkkcqQVJBHUS2l1Pwo2fRGPp0Q/wD1/TFuy3aiJ10KkqwKkcgiiJyRCaPAeKOlqI4FlDdHAPofSZ4gcAqdiIGv2Z7PbxDlEKqQrOS5IFAgdATdkYqWe0vZOp4et5TJIG1r75ogHbjB4Ms9keNPg9Uu6EkoV24umKmmB4wOOciXe1PaC+IVURXBDuwDbfNd0cHL1QrrWMnO5MfHd7bkx8d3t40A1kYIyPj3nCJf4rVV3LKgQGvKOBQA7D+n+sww5qNag1kk5rkgW1m7Y+YHPEqlrHyJ+t/8On/pKoGpBWg56Nq6NepRNfcB8PtEv9S95lnSxoCzQuheATzQ6cD6SenpWCxO1Rgt6/hUfeb0+tDMCKIWNAWee1DqSTgD1OJcNUaZ8htx/wCoLFH8nUfqOe23rDU1cbVG1cY6sRwWPU+nA+pNUDQTpuP+m3YC0PqAMp8AGHbaMTg0kHOoCPyK5P0YKP3laaTMGIF7RubIFLYF55yRxIQNg1w4GmFIW/Jm2D598mgVycYC2SOW31DwpYqEIfewQbbHnPA8wHeUo5UgjkZl66isNjAIMkFbIDd2BskVjHF2Acg31pfSnV0yjMrYKllI7MDRH1EjOuhU0RRFfTkEEYIIyCMGckQiIgJLedu3oTZ9ewPoM/UyMQLkYMm0kKVJKk3VH3lwDWQCP4u8Oyqu1TuuizZANcKLzV5zya7ZpiXa7S0n2srDJUg/Q3UuOkgN7gU6AHzkdFI+6e5OO14vPEbFj6pbFKB02qBXpfJ+ZMriajpDT9/L5/u8jb/7hxR/IM99vBiPovYng9P7FW2qxcHcSAepFZ6CfO+PRE1nC5QMMA9MFgCfWxK18U4unZQ3IU7R9BgfKQ09TbuwDuVl8wurxY7N6z0cvNjlhMZNWPLw8GeHJcsruVZ4xkLk6alUpaB70LNFmIF3jcf8hRETzvUREQERED7D2x4VdTTYkeZVLKeoIF18DU+Pnt+0/bQ1EKICA2GZq47Aes8Sev5meGee8P8AXz/+dxcnHxWZ+vfqfoiInkfQIiICJ1ELEBQWJ4ABJPwAljeGYc7P59O/mN2PnAqiS1NNkNMCp9e3cdx6y3R0LG5zsTOeSxH3UH3j68DqeAQr0dFnNDoLJOAq/iYnAH/YcmWO6phCT0LkVf6AcqPXk+lkTmrr2AqjYgztu7P4mONzZOenQASmAiS09NnNKpY9lBJ+gkYCIiBcutYCv5lHH4lH5T2/Lxk8HMauhtG4Hel1uHQ/hcfcb0ODRokC5TNPgdU6bFxyqt25NKtg4YBipo2DUDjaa6Zp7ZxygNBfR2o+buo47gggQ+0UnKKB+VnB+RZj/QyT7GBYeRuqmyD6oxs/wt9TwJ6vgX0003fypqAkGroA9u5FEZzfSGbljLJb30iPDb8pbfiBoFPVjdbPz4HfbYsNRdP3PM+POeF/QD1/Oc9gKs9TxQU0oKofeF5deDuPXBNDgdM5Od02krzRI+hqGnCf9+ss0tLdZJCqK3MeBfAock5x6HoCZXND6ZLJpjpX8zAFj8hQ/hhnK6PtBqYc7WqhqVz2D1z+rJHWxxU2kyttKncaoc3fFV7wPQiwZbY1CfuIgJwLIGAPizGsnv0AqfU/8IeAAT7Zss24Jf3UBINdrbd/u5ZNuHyPkzg47lZ7/X9rw09i+JKV9i3IYWUBqiCKLX+Hp0nm62k2mxV1KMOQwo/+J+o3PM9u+z119JsedAWRutjO2+xqvoektxfO4v8ApZ+euTGSb1dfi18KukFAZ+uQgwxHQt+Ff3PTB3BqKzjfghR7q/cW+NvRc8ixnJszi8b9wYg2wYXyfeN2GB69bPznWbaVdBtuxV2AwrcM8qQwwfxVmZfWuV3qKIk9ZNrEDilI+DAMP2IkIbl3NrvtVJBZAQF20vkzVBzzZvJ7ymIhSIiBq8Gh1WTSNeZgqsb8lnPHK5J298irN6vH+xjpIX3qwG3gG/N3GdtHGavpcyaO7SKamAQQyrZDEchse6O183YsZmvx3thtZWVkC7ijFrPvKKLHGbsnvk9zffD6/C+Xf4efP7fsnj1+XlxLNfRbTYqwo4PIIIOQykYZSMgjBlc4PQREQEREBJ6Oi2odqizk9AAByxJwqjqTgSej4csC5OxFIBY5834VH32rNfMkDMlreIBGxBsTGOWYjhnbqew90dBdkhP7ZdLGmdz1R1MjbfI0wcjtvOewXk5IiAiIgIiICIiAiIgIiICIiAhVJIAFk4AHU9BEt8N76YvzLj1sUIE/EuEB0190YdgffYHJv8APA+fJlzex/EABjouAeDXIoH+hB+cwHj5T9RXXsadOVDFFcllJBoUwNeXBaj02mV4vl/Iy4ZPGb2/PfCumnuXWVmAI/u+BuvzOWBBUgDgUWurAzKPGOxc7yCRQFYXYPdCAYCVkAd5f7bYnxGsSKJdrBuwe3mz9Zk1rsWK8qfTYu0/MUfnI9eGXljMv3EIiIaT0dZtM7kYoc5UkHPqI1mUsSq7FPC2TQ7WcmQiAiIgJd4ZQzbfxgqP18oPmwA+dymIDg5HByDY45B6iet4/28+umxkQLitoa1I4I82O3zmE6qOPOGDD760S3bepIs/mBB7hjmc+zTrqWPyoxb6NtH7w55cWOWUys9zr+I+G097AH3RlyOiDLH6cetDrKifSvTt6S59YbdiDapNmzbNXG49hztGPiQDK9NA27zBdoJz1Iryiuuf2h0Rmt9Ta6ahyGCk1zgBXXPXn+Yd5kliaxVWXBVuh6N0cdm9frcM5Y7WgDT3ox3K6imSjwbRqPqKINEWe0+v/AOEvFB9AJfm0ywI9GJZT8MkfIz41dMKLf5IPePqfwj9z2o3J+H8c+m4dCFIxSjG38JHUfGz88yy6eb5Xxfu47N++/wDY/SdNPKobJAFn17zJ47XGhoOzH3VNerG9o+pE8Vv+J3Vgh01Y0LYEgXXmIB6Cj16GeF7Y9o6ms5DsCEZgqrhRmrA6n1M6ZZeninwubPLXJ1vd/umXTYBCosu5VeMBAQcdyWA/l9ZPW0z5NMZYE3RBG96BWxjAVQT3DTNLNPWKqwAALYLddtUVHQA9evTicn1Lje474lgXNGwNqg9wqhQfmBcqiW+GA3qSNwFsQeoUFivzAr5w3JqaTOkqAF7LEAhFNUpyC7UasZCgXRvGLgdVf+mg+B1P83lbuSSSbJJJPcnJM3+J9lPp6GnrHh+R+EH3CfiP6iGcs8cLJle/UZ00BqY0wd2fIck9SVP3qGSKsAE5AJBWXT42u/f3kX9PR2/NlR0vDAPE7K+ztKrzfeY857LY93jveKr8QAGO3AOQOysNwX5A18obVsxJJJJJyScknuT1iIgadHXBXY9lMlSMshOSV7qTkocHkUcyrX0DpkA0QRasptWXjcp7YODRFEEAgiVzR4bXCgo4Loxsgcq3G9CeHwPRgKPQgM8S7xGhsqiHVrKOMBgOccqw6qePUEE16emzsFUFmY0ABZJ7AQIzSugqANqXmiumDTMOQzH7iH+ZulA7hK10uNr6nfDIh/L0d/XKjpuNEZXYkkkkkkkkkkknkknk+sCevrM5tumAAKVR+FQMAf15NkkyuIgIiICIlv8AZ260v6iFPxo5/aWS3pLlJ2qiXHwz80KPDbhtPzur9OZU+mVNEV/p3HcRZZ3EmWN6rkREjRERARLF8O5AaqBDMCxChgpptpat1HFCzGposnvKR7vw8wsZ4yL+hgVxE6i2QMZIGTQzjJ6CByAfl8JZ4jROmxU0SK4vsD1AIOeCARxzK4Gp9M6tugs8ug5B+86ryUJzj3eDiicdD0kgaojBGQexHBE0f2zU/Gx9Sbb+Y5/eBzR0AAGfypyOhevup/QtwPjQNWrqF2LHkm8cfAdgOAJx3LEliWJ5JNk/EmcgIiICIiAiW/2d9u7adtXdfd/F8PXiVQksvRE0OF06BUO+CdxYBbFhaUglq5s446XIa6DysuFa8HNMPeW+vQ/BhCTKVVEs09EsCcADBLEAX2zyfQSthRI7diCPqOYXc3olw8S1qWp9q7AHFgKAQBXpeJSBLk0hQZztU8Ae83TyjoLFbjjnkioVXpoWNDPU8AAdyTgD1MsLqvu+ZsefIo/kHPzOfQSOpq2KA2qMhR37k/ePqflQxK4Ayzw626iryLHp1/aVy7w77SWHKqa+J8uf5pZ2sTLiirHJ3W13TE8Duvfv8s1+JUhs80v+EAn6gyf2II35CDkdbxgHtkebpec1u1+N8F/dJq7191RsAOFJJUA3kiwP8zL42yrq2PMiImWSWaBUMu7C3TEchTgkeoBJlcQJaumUJVhkYP8AqO4PIPW5u1/bPiNRSjPasKI2IBXyXEzprKQFdSQPdZTTqO2cOvoaPYgSOzT/ABv/APGv/wC4Zywxystkuut/hXpaZdgo5OBfA7knoALJPQAzusV3Ns92ztvnb0v1qWNrAAqgKhsMxyzDtjCrxgfMnEohoiIgIiIF/hm94YOLVWPlLg1ZBNEhS1X/ANpLW1ClhSF3qA4Sq5PkscKRtJUGr+AAzRLv0u/TraZWrBWwCLBFqeCL5HrOSerrM+3cd21Qovoouh+5+shIhERAQqkkAZJwB69olqNtUnqfKPQfeP0IHzPaWRMrqenS4TC5bq/r2TsPzcn0EqWrzkXnuR1+c5EW7SY6X+J1Q1AHdRY3VDNAKo6AAD95Vv8ALt6XY9O9fHH0EjEtytuzHGSaIl3ifEHU22ANiqgoVhbon1zKZloiIgGNkk5JyScknuT1M0eGV12kAbWY4dlVHKeYhgxAYC/3xkzPNFI+22VDhW8jVtVcPa3uY5BwM0epICHiNLYVwwDIjruUrasMkXyu7cA3Xb8pMPp/Z1sP2m73rNbO3NfLbfXd92F8QMDafs/JuTdztskqxB2EsztgY3kZF3S6bTVhsKbU2MgGviLojuDAjERAREQEREBERATqEAgkbgCCR3HaciBYuswffdtd2ep6g+hGK7Ylg+zB3C2HI0yPoGa8qPTJ9JniGbjK67EkkmySST3JyTLNPUABVhuUmxRohuLBo8jkV0HaVRC3GWaT1dTdQApVwBd88knqT1Pw4AAEIiCTXqJablTY+HUY+INyLGyT3JMRCkREBNGmlIzHjcoxycE7b6crmZ5X4rWZNlGsM3p5iVIIOCKQSxYsfxJ3XxWBXAXPlAPTJx1s95pc7tMEGhuwtmhYNlc+7Yx8x6nzzpgDdXmAs6Z6Dnec2U67ee+MmXg9VmL2btb+YZf6KWx0lVbERMskREC0a9IU2plt27b5xitoa/d9PWVREBERAREQEREBERAREQEREBERAREQEREBERAREQEREBNXiP8Al6fodUD0Fpj9z9YiBliIgIiICIiAiIgIiICIiAiIgIiICIiAiIgJzU/56fpT/DETUWPNVzd2bu7vN3zc3MK1yBgf3uB+h4iQi2IiRCIiAiIgIiICIiAiIgIiICIiAiIgIiICIiAiIgIiICIiB//Z')
st.sidebar.header('User Input Parameters')

def user_input_features():
    voice_mail_plan = st.sidebar.selectbox('voice_mail_plan',('0','1'))
    voice_mail_messages = st.sidebar.number_input("Insert voice_mail_messages",min_value=0,max_value=100)
    day_mins =  st.sidebar.number_input ("insert day_mins",min_value=0,max_value=500)
    evening_mins = st.sidebar.number_input("insert evening_mins",min_value=0,max_value=500)
    night_mins =  st.sidebar.number_input("insert night_mins",min_value=0,max_value=500)
    international_mins = st.sidebar.number_input("insert international_mins",min_value=0,max_value=500)
    customer_service_calls = st.sidebar.number_input("insert customer_service_calls",min_value=0,max_value=500)
    international_plan = st.sidebar.selectbox('international_plan',('0','1'))
    day_calls = st.sidebar.number_input("insert day_calls",min_value=0,max_value=500)
    day_charge = st.sidebar.number_input("insert day_charges",min_value=0,max_value=500)
    evening_calls = st.sidebar.number_input("insert evening_calls",min_value=0,max_value=500)
    evening_charge = st.sidebar.number_input("insert evening_charge",min_value=0,max_value=500)
    night_calls = st.sidebar.number_input("insert night_calls",min_value=0,max_value=500)
    nighr_charges = st.sidebar.number_input("insert night_charge",min_value=0,max_value=500)
    international_calls = st.sidebar.number_input("insert international_calls",min_value=0,max_value=500)
    ingternational_charge = st.sidebar.number_input("insert international_charge",min_value=0,max_value=500)
    total_charge = st.sidebar.number_input("insert total_charge",min_value=0,max_value=500)

    data = {'voice_mail _plan':voice_mail_messages,
            'voice_mail_messages':voice_mail_messages,
            'day_mins':day_mins,
            'evening_mins':evening_mins,
            'night_mins': night_mins,
            'international_mins':international_mins,
            'customer_service_calls':customer_service_calls,
            'international_plan':international_plan,
            'day_calls': day_calls,
            'day_charge': day_charge,
            'evening_calls':evening_calls,
            'evening_charge':evening_charge,
            'night_calls': night_calls,
            'nighr_charges':nighr_charges,
            'international_calls':international_calls,
            'ingternational_charge':ingternational_charge,
            'total_charge' :total_charge}
    features = pd.DataFrame(data,index = [0])
    return features

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

churn= pd.read_csv("telecommunications_churn (1) (1).csv")
churn.drop(["voice_mail_messages"],inplace=True,axis = 1)
churn= churn.dropna()

X = churn.drop(['churn'],axis=1)
Y = churn['churn']
clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('NO-churn' if prediction_proba[0][1]>0.698 else 'churn')

st.subheader('Prediction Probability')
st.text('''O-->(churn)
1-->(No churn)''')
st.write(prediction_proba)

