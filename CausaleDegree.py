import math
from datetime import datetime,timedelta
class CausaleDegree:
    def __init__(self):
        self.DI = 0
        self.periode = 12
        self.rate_E = 0.9

    def DI_causal(self, E1_cause, E2_effect):
        i = 0
        sum = 0
        j = 0
        
        if E2_effect.pos_dates[0] <= E1_cause.pos_dates[0]:
            i = 1
        
        while i < len(E2_effect.pos_dates) and j < len(E1_cause.pos_dates):
            if E2_effect.pos_dates[i] > E1_cause.pos_dates[j]:
                sum += math.exp((-E2_effect.Event_Mu(E2_effect.pos_dates, self.rate_E, self.periode)) * (E2_effect.pos_dates[i] - E1_cause.pos_dates[j]))
                i += 1
                j += 1
            elif E2_effect.pos_dates[i] <= E1_cause.pos_dates[j]:
                sum += math.exp((-E2_effect.Mu) * (E2_effect.pos_dates[i] - E1_cause.pos_dates[j - 1]))
                i += 1
                j += 1
                
        if i > (j - 1):
            pos = i
            while pos < len(E2_effect.pos_dates):
                sum += math.exp((-E2_effect.Mu) * (E2_effect.pos_dates[pos] - E1_cause.pos_dates[j - 1]))
                pos += 1

        self.DI = sum / E2_effect.Event_Occurence(E2_effect.pos_dates)
        print(E1_cause.ID_e + " causes " + E2_effect.ID_e + " with an Influence Degree of " + str(self.DI))
        return self.DI

    def DI_causal_2(self, E1_cause, E2_effect):
        i = 0
        sum = 0
        j = 0
        t = 0

        if E1_cause.Measure == E2_effect.Measure:
            self.DI = 0
        else:
            E1_pos_dates = sorted(E1_cause.pos_dates, reverse=True)
            E1_pos_dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in E1_pos_dates]
            
            E2_pos_dates = sorted(E2_effect.pos_dates, reverse=True)
            E2_pos_dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in E2_pos_dates]
            print("e2postdates",E2_pos_dates)
            end1 = len(E2_pos_dates)                
            T0 = 0

            if end1 == 1:
                # T0 = E2_pos_dates[0] ?????
                T0=1
            else:
                T0 = (E2_pos_dates[0] - E2_pos_dates[end1 - 1]) / (end1 - 1)          
                print("T0",T0)
                
                T0=self.convertToDate(str(T0))
                print("T0 changed",T0)
            Mu = (-math.log(1 - self.rate_E)) / T0

            for i in range(len(E2_pos_dates)):
                while t == 0 and j < len(E1_pos_dates):
                    if E2_pos_dates[i] > E1_pos_dates[j]:
                        # sum += math.exp((-Mu) * ( self.convertToDate(str( E2_pos_dates[i] - E1_pos_dates[j] )) ) )
                        ti=E2_pos_dates[i] - E1_pos_dates[j]
                        print("time",ti)
                        ti=self.convertToDate(str(ti))
                        sum += math.exp((-Mu) * (ti))
                        t = 1
                    else:
                        j += 1
                t = 0

        self.DI = sum / end1 if end1 != 0 else 0
        return self.DI
        
    def convertToDate(self, duration_str):
        # Parse the duration string into a timedelta object
        duration = timedelta(days=0, seconds=0, microseconds=0)
        parts = duration_str.split(", ")
        for part in parts:
            if "days" in part:
                duration += timedelta(days=int(part.split()[0]))
            elif "day" in part:
                duration += timedelta(days=int(part.split()[0]))
            else:
                time_parts = part.split(":")
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds = 0
                if len(time_parts) > 2:
                    seconds = int(time_parts[2].split(".")[0])
                microseconds = 0
                if len(time_parts) > 2 and "." in time_parts[2]:
                    microseconds = int(time_parts[2].split(".")[1])
                duration += timedelta(hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds)

        # Convert the timedelta object to days
        total_days = duration.days + duration.seconds / (24 * 3600) + duration.microseconds / (24 * 3600 * 10**6)
        print("Total days:", total_days)

        return total_days
