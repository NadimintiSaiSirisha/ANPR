def getAccuracy(labelPred, Output):
  count = 0
  #print("labelPred: " ,labelPred)
  #print("type(labelPred): ",type(labelPred))
  #print("Output", Output)
  #print("Type(Output): ", type(Output))
  labelsActual = Output.split("-")[4].split("_")
  #print("labelsActual: ", labelsActual)
  #print("type(labelsActual): ", type(labelsActual))
  for i in range(7) :
    #print("Actual: ", type(labelsActual[0]))
    #print("LabelPred: ", type(labelPred[0]))
    if labelPred[i]==int(labelsActual[i]):
      count+=1
    else:
      break
  #print(" ", count,"/7 ")
  if count==7 :
    return True
  else:
    return False
