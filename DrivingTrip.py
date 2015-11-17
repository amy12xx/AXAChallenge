# Create a class to hold all trip parameters
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

'''
Following features are extracted from the GPS coordinate data

SPEED:
1. Standard deviation of speed
2. Maximum speed
3. Average speed

ACCELERATION:
1. Average Acceleration
2. Average sudden acceleration or jerk (differential of acceleration)
3. high acceleration time/ running time
2. high deceleration time / running time 
3. Average Positive Acceleration
4. Std. dev. of Positive Acceleration
6. Average Deceleration
7. Std. dev. of Deceleration 

RUNNING/STOP TIME:
1. low speed time (%) : percent of time vehicle is in low speeds
2. Percent idle (%): percent of time vehicle is stopped
3. # of stops per traversed distance

ANGLE:
1. Std deviation of Slip angle
2. Avg slip angle
3. Max slip angle

INTERVALS:
1. percent of time in low, mid and high speed interval
2. percent of time over FTP limit
3. slip over threshold by total time
4. slip over threshold by total distance

5-SECOND INTERVAL:
1. max and avg slip angle
2. max and avg speed
3. avg sudden acceleration
'''

HIGH_ACCELERATION = 1.1
HIGH_DECELERATION = -1.1
LOW_SPEED = 2
STOP_SPEED = 0.4
LOW_SPEED_INTERVAL = 30.0
MID_SPEED_INTERVAL = [31.0, 44.0]
FTP_CITY_LIMIT = 21.2
HIGH_SPEED_INTERVAL = 44.1
HIGH_SLIP = 2.3

class DrivingTrip:
    def __init__(self, X, trip_num):
        self.X = X
        self.trip_num = trip_num
        self.euclidean_dist = 0.0

        self.inst_speed = np.zeros((X.shape))
        self.max_speed = 0.0
        self.avg_speed = 0.0
        self.std_dev_speed= 0.0
        
        self.inst_acceleration = np.zeros((X.shape)) 
        self.avg_acceleration = 0.0
        self.avg_sudden_acceleration = 0.0
        self.high_acceleration_time = 0.0
        self.high_deceleration_time = 0.0
        self.avg_positive_acceleration = 0.0
        self.std_dev_positive_acceleration = 0.0
        self.avg_deceleration = 0.0
        self.deceleration_std_dev = 0.0
        
        self.low_speed_time = 0.0
        self.total_trip_time = 0.0
        self.total_stop_time = 0.0
        self.total_run_time = 0.0
        self.idle_time_percent = 0.0
        self.low_time_percent = 0.0
        self.stop_time_by_traversed_distance = 0.0
        
        self.slip_angle_arr = np.zeros((X.shape))
        self.max_slip_angle = 0.0
        self.avg_slip_angle = 0.0
        self.slip_angle_std_dev = 0.0
        
        self.speed_for_low_time_interval = 0.0
        self.speed_for_mid_time_interval = 0.0
        self.speed_for_high_time_interval = 0.0
        self.speed_over_ftp_city_limit = 0.0
        self.slip_over_interval_by_total_dist = 0.0
        self.slip_over_interval_by_total_time = 0.0

        self.avg_slip_angle_5s, self.max_slip_angle_5s, self.max_speed_5s, self.avg_speed_5s, \
            self.sudden_acceleration_5s = 0.0, 0.0, 0.0, 0.0, 0.0

    def nan_helper(self, y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def fit(self):
        # assign all parameters based on X
        self.max_speed = self.calc_max_speed()
        self.avg_speed = self.calc_avg_speed()
        self.std_dev_speed = self.calc_speed_std_dev()

        self.avg_acceleration = self.calc_avg_acceleration()
        self.avg_sudden_acceleration = self.calc_sudden_acceleration()
        self.high_acceleration_time = self.calc_high_acceleration_time()
        self.high_deceleration_time = self.calc_high_deceleration_time()
        self.avg_positive_acceleration = self.calc_avg_positive_acceleration()
        self.std_dev_positive_acceleration = self.calc_std_dev_positive_acceleration()
        self.avg_deceleration = self.calc_average_deceleration()
        self.deceleration_std_dev = self.calc_std_dev_deceleration()

        self.idle_time_percent = self.calc_idle_time_percent()
        self.low_time_percent = self.calc_low_time_percent()
        self.stop_time_by_traversed_distance = self.calc_stops_by_traversed_dist()

        self.max_slip_angle = self.calc_max_slip_angle()
        self.avg_slip_angle = self.calc_avg_slip_angle()
        self.slip_angle_std_dev = self.calc_std_dev_slip_angle()

        self.speed_for_low_time_interval = self.calc_speed_low_time_interval()
        self.speed_for_mid_time_interval = self.calc_speed_mid_time_interval()
        self.speed_for_high_time_interval = self.calc_speed_high_time_interval()
        self.speed_over_ftp_city_limit = self.calc_speed_over_ftp_city_limit()
        self.slip_over_interval_by_total_dist = self.calc_slip_over_interval_by_total_dist()
        self.slip_over_interval_by_total_time = self.calc_slip_over_interval_by_total_time()

        self.avg_slip_angle_5s, self.max_slip_angle_5s, self.max_speed_5s, self.avg_speed_5s, \
            self.sudden_acceleration_5s = self.calc_for_5_second_interval()

    def calc_euclidean_dist(self):
        distance = self.calc_inst_speed()
        self.euclidean_dist = distance[distance.shape[0]-2] - distance[0]
        return self.euclidean_dist

    def calc_inst_speed(self):
        # Instantaneous speed as hypothenuse of (X1,Y1) and (X2,Y2)
        X = self.X['x']
        X_diff = np.diff(X)

        Y = self.X['y']
        Y_diff = np.diff(Y)
        
        speed = np.hypot(X_diff, Y_diff) # speed is equal to distance since t2-t1 is always 1 in our data set
        self.inst_speed = speed
        return self.inst_speed

    def calc_max_speed(self):
        # Calculate the maximum speed reached in 1 second
        speed = np.array(self.calc_inst_speed())
        self.max_speed = speed.max(axis=0)
        return self.max_speed
    
    def calc_avg_speed(self):
        # Calculate the average speed reached in 1 second
        speed = np.array(self.calc_inst_speed())
        self.avg_speed = np.average(speed)
        return self.avg_speed    

    def calc_speed_std_dev(self):
        self.std_dev_speed = np.std(self.calc_inst_speed(), axis =0)
        return self.std_dev_speed

    def calc_inst_acceleration(self):
        #Acceleration is defined as rate of change of velocity
        velocity = np.array(self.calc_inst_speed())
        velocity_diff = np.diff(velocity)
        self.inst_acceleration = velocity_diff
        return self.inst_acceleration

    def calc_avg_acceleration(self):
        # Average of acceleration 
        acceleration = np.array(self.calc_inst_acceleration()) 
        self.avg_acceleration = np.average(acceleration)
        return self.avg_acceleration

    def calc_sudden_acceleration(self):
        # Sudden acceleration or jerk defined as rate of change of acceleration
        acceleration = np.array(self.calc_inst_acceleration()) 
        acc_diff = np.diff(acceleration)
        change_in_acc = np.average(acc_diff)
        self.avg_sudden_acceleration = change_in_acc
        return self.avg_sudden_acceleration

    ########## DO BOXPLOT OF ACCELERATION TO CHECK INTERVAL        
    def calc_high_acceleration_time(self):
        inst_acc = self.calc_inst_acceleration()
        inst_acc = inst_acc[inst_acc >= HIGH_ACCELERATION]
        self.high_acceleration_time = float(inst_acc.shape[0]) / float(self.calc_total_run_time())
        return self.high_acceleration_time
        
    def calc_high_deceleration_time(self):
        inst_acc = self.calc_inst_acceleration()
        inst_acc = inst_acc[inst_acc <= HIGH_DECELERATION]
        self.high_deceleration_time = float(inst_acc.shape[0]) / float(self.calc_total_run_time())
        return self.high_deceleration_time
        
    def calc_avg_positive_acceleration(self):
        inst_acc = self.calc_inst_acceleration()
        positive_acceleration = inst_acc[inst_acc > 0]
        self.avg_positive_acceleration = np.average(positive_acceleration)
        return self.avg_positive_acceleration
        
    def calc_std_dev_positive_acceleration(self):
        pos_acc = self.calc_inst_acceleration()
        pos_acc = pos_acc[pos_acc > 0]
        self.std_dev_positive_acceleration = np.std(pos_acc)
        return self.std_dev_positive_acceleration

    def calc_average_deceleration(self):
        inst_acc = self.calc_inst_acceleration()
        deceleration = inst_acc[inst_acc < 0]
        self.avg_deceleration = np.average(deceleration)
        return self.avg_deceleration
    
    def calc_std_dev_deceleration(self):
        inst_acc = self.calc_inst_acceleration()
        deceleration = inst_acc[inst_acc < 0]
        self.deceleration_std_dev = np.std(deceleration)
        return self.deceleration_std_dev

    def calc_low_speed_time(self): 
        # Calculate amount of time spent at low speeds (at threshold)
        X = np.array(self.calc_inst_speed())
        X = X[X < LOW_SPEED]
        self.low_speed_time = float(X.shape[0])
        return self.low_speed_time

    def calc_total_stop_time(self):
        X = np.array(self.calc_inst_speed())
        X = X[X < STOP_SPEED] 
        self.total_stop_time = float(X.shape[0])
        return self.total_stop_time

    def calc_total_trip_time(self):
        # Total trip time, including stoppage time
        self.total_trip_time = float(self.X.shape[0])
        return self.total_trip_time

    def calc_total_run_time(self):
        self.total_run_time = self.calc_total_trip_time() - self.calc_total_stop_time()
        return self.total_run_time

    def calc_idle_time_percent(self):
        self.idle_time_percent = float(100 * (float(self.calc_total_stop_time())/float(self.calc_total_trip_time())))
        return self.idle_time_percent

    def calc_low_time_percent(self):
        self.low_time_percent = float(100 * (float(self.calc_low_speed_time())/float(self.calc_total_trip_time())))
        return self.low_time_percent

    def calc_stops_by_traversed_dist(self):
        total_traversed_dist = self.calc_inst_speed().sum()
        self.stop_time_by_traversed_distance = (float(self.calc_total_stop_time()) / float(total_traversed_dist))
        return self.stop_time_by_traversed_distance     

    def calc_slip(self):
        # First find angle Phi between two distances from reference axis, then slip angle is difference of those two angles.
        Y = self.X['y']
        Y_diff = np.diff(Y)

        speed = self.calc_inst_speed()

        phi_angle = np.arcsin(np.divide(Y_diff, speed))

        # handle NaN values
        nans, x= self.nan_helper(phi_angle)
        phi_angle[nans]= np.interp(x(nans), x(~nans), phi_angle[~nans])

        self.slip_angle_arr = np.diff(phi_angle)
        return self.slip_angle_arr    

    def calc_max_slip_angle(self):
        slip_angle_arr = self.calc_slip()
        self.max_slip_angle = np.percentile(slip_angle_arr,95)
        return self.max_slip_angle 

    def calc_avg_slip_angle(self):
        slip_angle_arr = self.calc_slip()
        self.avg_slip_angle = np.average(slip_angle_arr)
        return self.avg_slip_angle 
        
    def calc_std_dev_slip_angle(self):
        slip_angle_arr = self.calc_slip()
        self.slip_angle_std_dev = np.std(slip_angle_arr)
        return self.slip_angle_std_dev   
        
    def calc_speed_low_time_interval(self): 
        speed = self.calc_inst_speed()
        low_speed = speed[speed < LOW_SPEED_INTERVAL]
        self.speed_for_low_time_interval  = float(100 * float(float(low_speed.shape[0]) / float(self.calc_total_run_time())))
        return self.speed_for_low_time_interval

    def calc_speed_mid_time_interval(self):
        speed = self.calc_inst_speed()
        mid_speed = speed[(MID_SPEED_INTERVAL[0] < speed) & (speed < MID_SPEED_INTERVAL[1])]
        self.speed_for_mid_time_interval  = float(100 * float(float(mid_speed.shape[0]) / float(self.calc_total_run_time())))
        return self.speed_for_mid_time_interval

    def calc_speed_high_time_interval(self):
        speed = self.calc_inst_speed()
        high_speed = speed[speed > HIGH_SPEED_INTERVAL]
        self.speed_for_high_time_interval  = float(100 * float(float(high_speed.shape[0]) / float(self.calc_total_run_time())))
        return self.speed_for_high_time_interval

    def calc_speed_over_ftp_city_limit(self):
        speed = self.calc_inst_speed() / 1609.34 # to convert meters to miles
        high_speed = speed[speed > FTP_CITY_LIMIT]
        self.speed_over_ftp_city_limit  = float(100 * float(float(high_speed.shape[0]) / float(self.calc_total_run_time())))
        return self.speed_over_ftp_city_limit        

    def calc_slip_over_interval_by_total_dist(self):
        slip = self.calc_slip()
        slip = slip[np.absolute(slip) > HIGH_SLIP]
        self.slip_over_interval_by_total_dist = float(float(slip.shape[0]) / float(self.calc_inst_speed().sum()))
        return self.slip_over_interval_by_total_dist

    def calc_slip_over_interval_by_total_time(self):
        slip = self.calc_slip()
        slip = slip[np.absolute(slip) > HIGH_SLIP]
        self.slip_over_interval_by_total_time = float(float(slip.shape[0]) / float(self.X.shape[0]))
        return self.slip_over_interval_by_total_time

    def calc_for_5_second_interval(self):
        # To calculate slip over 5 seconds, first calculate differences over 5 seconds, then -arctan((y2-y1)/(x2-x1)), then difference of that angle
        X = np.array(self.X)

        X1 = X[4::,:]
        X2 = X[0::1,:]
        diff = X1 - X2[:-4,:]

        angle = -np.arctan(np.diff(diff[:,0]), np.diff(diff[:,1]))
        slip_angle = np.diff(angle)
        self.avg_slip_angle_5s = np.average(slip_angle)
        self.max_slip_angle_5s = np.max(slip_angle)

        # To calculate traversed distance for calculating speed in 5s intervals
        inst_speed = self.calc_inst_speed()

        speed1 = inst_speed[4::]
        speed2 = inst_speed[3::]
        speed3 = inst_speed[2::]
        speed4 = inst_speed[1::]
        speed5 = inst_speed[0::]

        speed = (speed1 + speed2[:-1] + speed3[:-2] + speed4[:-3] + speed5[:-4]) / 5

        self.max_speed_5s = np.percentile(speed, 95)
        self.avg_speed_5s = np.average(speed)

        self.sudden_acceleration_5s = np.percentile(np.diff(speed), 95)

        return self.avg_slip_angle_5s, self.max_slip_angle_5s, self.max_speed_5s, self.avg_speed_5s, self.sudden_acceleration_5s

def main():
    path = 'D:/python_files/connected_cars/data/drivers/drivers'
    driver = 10
    trip_num = 136
    path = path + '/' + str(driver) + '/' + str(trip_num) + '.csv'
    X = pd.read_csv(path)

    dt = DrivingTrip(X, trip_num)
    dt.fit()

    print 'Max speed: %0.3f' % dt.max_speed
    print 'Average speed: %0.3f' % dt.avg_speed
    print 'Standard Deviation Speed %0.3f' % dt.std_dev_speed
    print
    print 'Average acceleration: %0.3f' % dt.avg_acceleration
    print 'Avg sudden acceleration: %0.3f' % dt.avg_sudden_acceleration
    print 'High Acceleration: %0.0f' %dt.high_acceleration_time
    print 'High Deceleration: %0.3f' %dt.high_deceleration_time
    print 'Average Positive Acceleration: %0.3f' %dt.avg_positive_acceleration
    print 'Standard Deviation of Positive Acceleration: %0.3f' %dt.std_dev_positive_acceleration
    print 'Avg deceleration: %0.3f' %dt.avg_deceleration
    print 'Std deviation deceleration: %0.3f' %dt.deceleration_std_dev
    print
    print 'Idle time Percentage: %0.3f' % dt.idle_time_percent
    print 'Stop time percentage: %0.3f' %dt.low_time_percent
    print 'Stops by traversed distance: %0.3f' %dt.stop_time_by_traversed_distance
    print
    print 'slip angle max: %0.3f' % dt.max_slip_angle
    print 'slip angle avg: %0.3f' % dt.avg_slip_angle
    print 'Std dev slip angle: %0.3f' %dt.slip_angle_std_dev
    print    
    print 'Percent speed spent in low time interval %0.3f' %dt.speed_for_low_time_interval
    print 'Percent speed spent in mid time interval %0.3f' %dt.speed_for_mid_time_interval
    print 'Percent speed spent in high interval %0.3f' %dt.speed_for_high_time_interval
    print 'Percent speed over FTP city driving limit %0.3f' %dt.speed_over_ftp_city_limit
    print 'slip_over_interval_by_total_dist %0.5f' %dt.slip_over_interval_by_total_dist
    print 'slip_over_interval_by_total_time %0.5f' %dt.slip_over_interval_by_total_time
    print
    print 'Max speed 5s interval: %0.3f' %dt.max_speed_5s
    print 'Avg speed 5s interval: %0.3f' %dt.avg_speed_5s
    print 'Max slip angle 5s interval: %0.3f' %dt.max_slip_angle_5s
    print 'Avg slip angle 5s interval: %0.3f' %dt.avg_slip_angle_5s
    print 'Sudden acceleration 5s interval: %0.3f' %dt.sudden_acceleration_5s
    
if __name__ == '__main__':
    main()
