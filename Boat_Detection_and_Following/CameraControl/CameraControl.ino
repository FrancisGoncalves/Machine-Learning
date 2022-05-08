#include <SerCom0.h>
#include <Servo.h>

SerCom0 Controller; // Serial communication object
Servo top_servo; // Top servo to control
Servo bottom_servo; // Bottom servo to control

int center_point_microseconds = 1500;
float command_x = 0;
float command_y = 0;
float Kp_x = 0.1;
float Kp_y = 0.1;
float Kd_x = 0.5;
float Kd_y = 0.5;
int previous_error_x = 0;
int previous_error_y = 0;
int center_threshold_x = 10;
int center_threshold_y = 10;
int position_update_x = 0;
int position_update_y = 0;

void setup() {
  Serial.begin(115200);
  bottom_servo.attach(9);  // attaches the servo on pin 9 to the bottom servo object
  top_servo.attach(1); // attaches the servo on pin 1 to the top servo object
  top_servo.writeMicroseconds(center_point_microseconds);
  bottom_servo.writeMicroseconds(center_point_microseconds);
  //  delay(3000);
}

void loop() {
  //  Controller.get();
  //  int error_x = Controller.channel[2] - Controller.channel[0];
  //  int error_y = Controller.channel[3] - Controller.channel[1];
  // Get
  if (Serial.available() > 0) {
    int bb_center_y = Serial.parseInt();
    int bb_center_x = Serial.parseInt();
    int frame_center_y = Serial.parseInt();
    int frame_center_x = Serial.parseInt();
    int box_empty = Serial.parseInt();

    int error_x = frame_center_x - bb_center_x;
    int error_y = frame_center_y - bb_center_y;
    float dx_dt = (error_x - previous_error_x);
    float dy_dt = (error_y - previous_error_y);

    if (abs(error_x) > center_threshold_x && box_empty == 0) {
      position_update_x += 0.5 * error_x;
    }
    if (abs(error_y) > center_threshold_y && box_empty == 0) {
      position_update_y -= 0.5 * error_y;
    }


    //  if (error_x > 0 && error_x > center_threshold_x && box_empty == 0){
    //    position_update_x += 0.5*  abs(error_x);
    //  }
    //  if (error_x < 0 && error_x < -center_threshold_x && box_empty == 0){
    //    position_update_x -= 0.5* abs(error_x);
    //  }
    //   if (error_y > 0 && error_y > center_threshold_y && box_empty == 0){
    //    position_update_y -= 0.5*abs(error_y);
    //  }
    //   if (error_y < 0 && error_y < -center_threshold_y && box_empty == 0){
    //    position_update_y += 0.5*abs(error_y);
    //  }

    command_x = Kp_x * error_x; //+ Kd_x * dx_dt;
    command_y = Kp_y * error_y; //+ Kd_y * dy_dt;
    top_servo.writeMicroseconds(center_point_microseconds + position_update_y);
    bottom_servo.writeMicroseconds(center_point_microseconds + position_update_x);
    previous_error_x = error_x;
    previous_error_y = error_y;

    //  if (Serial.availableForWrite()) {
    //          Serial.println(error_y);
    ////          Serial.println(bb_center_y);
    ////          Serial.println(frame_center_x);
    ////          Serial.println(frame_center_y);
    //  }
  }
}
