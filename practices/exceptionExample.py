while True :
    try :
        x = int(input('please enter a number: \n'))
        break
    except ValueError :
        print('that was not a valid number. Try again')