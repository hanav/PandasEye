from dateutil.parser import parse
import pandas as pd

class DateUtils:
    def __init__(self, dateArray):
        self.outputDates = self.ConvertTimeofday(dateArray)

    def ConvertTimeofday(self, dateArray):
        outputDates = list()

        for dateItem in dateArray:
            converted = parse(dateItem)
            outputDates.append(converted)

        outputDates = pd.Series(data=outputDates)
        return outputDates