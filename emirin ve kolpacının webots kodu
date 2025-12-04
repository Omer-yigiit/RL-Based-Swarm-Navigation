#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include <webots/device.h>
#include <webots/distance_sensor.h>
#include <webots/led.h>
#include <webots/motor.h>
#include <webots/robot.h>
#include <webots/gps.h>
#include <webots/compass.h>
#include <webots/emitter.h>
#include <webots/receiver.h>

#define MAX_SPEED 6.28
#define TIME_STEP 64
#define COMMUNICATION_CHANNEL 1
#define DISTANCE_SENSORS_NUMBER 8

static WbDeviceTag distance_sensors[DISTANCE_SENSORS_NUMBER];
static double ps_values[DISTANCE_SENSORS_NUMBER];
static const char *ps_names[DISTANCE_SENSORS_NUMBER] = {
  "ps0", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6", "ps7"
};

static WbDeviceTag left_motor, right_motor;
static WbDeviceTag gps, compass, emitter, receiver;

#define FORMATION_SQUARE 0
#define FORMATION_TRIANGLE 1

typedef struct {
    double x, z, theta;
    int formation_type;
} LeaderPacket;

int my_id = -1;
double my_pos[3]; 

// Braitenberg Ağırlıkları
static double weights[8][2] = {
    {-1.5, -1.0}, {-1.5, -1.0}, {-0.5, 0.5}, {0.0, 0.0},
    {0.0, 0.0},   {0.5, -0.5},  {1.0, 1.5},  {1.0, 1.5}
};

int get_robot_id() {
    const char* name = wb_robot_get_name();
    int len = strlen(name);
    if (len > 0 && name[len-1] >= '0' && name[len-1] <= '9') return name[len-1] - '0';
    return 0; 
}

// Pusula Hesabı
double get_heading_rad() {
    const double *north = wb_compass_get_values(compass);
    if (!north) return 0.0;
    // Webots X ekseni Kuzey kabul edilir
    double angle = atan2(north[0], north[2]);
    return angle; 
}

void init_devices() {
    int i;
    for (i = 0; i < DISTANCE_SENSORS_NUMBER; i++) {
        distance_sensors[i] = wb_robot_get_device(ps_names[i]);
        if(distance_sensors[i]) wb_distance_sensor_enable(distance_sensors[i], TIME_STEP);
    }
    left_motor = wb_robot_get_device("left wheel motor");
    right_motor = wb_robot_get_device("right wheel motor");
    if(left_motor) { wb_motor_set_position(left_motor, INFINITY); wb_motor_set_velocity(left_motor, 0.0); }
    if(right_motor) { wb_motor_set_position(right_motor, INFINITY); wb_motor_set_velocity(right_motor, 0.0); }
    
    gps = wb_robot_get_device("gps");
    if(gps) wb_gps_enable(gps, TIME_STEP);
    compass = wb_robot_get_device("compass");
    if(compass) wb_compass_enable(compass, TIME_STEP);
    
    emitter = wb_robot_get_device("emitter");
    if(emitter) wb_emitter_set_channel(emitter, COMMUNICATION_CHANNEL);
    receiver = wb_robot_get_device("receiver");
    if(receiver) { wb_receiver_enable(receiver, TIME_STEP); wb_receiver_set_channel(receiver, COMMUNICATION_CHANNEL); }
}

int main(int argc, char **argv) {
    wb_robot_init();
    init_devices();
    
    my_id = get_robot_id();
    printf("--- Robot %d Baslatildi ---\n", my_id);

    double left_speed = 0, right_speed = 0;
    int current_formation = FORMATION_SQUARE;
    
    LeaderPacket leader_data = {0};
    bool leader_data_received = false;

    while (wb_robot_step(TIME_STEP) != -1) {
        
        // --- 1. ISINMA TURU (Kritik Düzeltme) ---
        // İlk 1 saniye robotlar hareket etmesin, GPS ve Pusula kendine gelsin.
        if (wb_robot_get_time() < 1.0) {
            wb_motor_set_velocity(left_motor, 0);
            wb_motor_set_velocity(right_motor, 0);
            continue;
        }

        // Sensör Okuma
        for (int i=0; i<8; i++) 
            ps_values[i] = (distance_sensors[i]) ? wb_distance_sensor_get_value(distance_sensors[i]) : 0;
        
        if(gps) {
            const double *vals = wb_gps_get_values(gps);
            if(vals) { my_pos[0] = vals[0]; my_pos[1] = vals[2]; } 
        }
        if(compass) my_pos[2] = get_heading_rad();

        // Engel Algılama
        double b_left = 0, b_right = 0;
        bool obstacle = false;
        for (int i=0; i<8; i++) {
            if (ps_values[i] > 80.0) {
                obstacle = true;
                b_left += ps_values[i] * weights[i][0];
                b_right += ps_values[i] * weights[i][1];
            }
        }
        // Engel tepkisini azalttık ki hemen panik yapmasınlar
        b_left *= 0.003; 
        b_right *= 0.003;

        // --- LİDER ---
        if (my_id == 0) {
            if (ps_values[0]>80 || ps_values[7]>80 || ps_values[6]>80 || ps_values[1]>80) 
                current_formation = FORMATION_TRIANGLE;
            else if(!obstacle) 
                current_formation = FORMATION_SQUARE;

            double base = MAX_SPEED * 0.5;
            left_speed = base + b_left;
            right_speed = base + b_right;

            if(emitter) {
                LeaderPacket pkt = { my_pos[0], my_pos[1], my_pos[2], current_formation };
                wb_emitter_send(emitter, &pkt, sizeof(LeaderPacket));
            }
        } 
        // --- TAKİPÇİLER ---
        else {
            if(receiver) {
                while(wb_receiver_get_queue_length(receiver) > 0) {
                    leader_data = *(LeaderPacket*)wb_receiver_get_data(receiver);
                    leader_data_received = true;
                    wb_receiver_next_packet(receiver);
                }
            }

            if (leader_data_received) {
                double dx = 0, dy = 0;

                // Formasyon Ayarları (Sağ/Sol Düzeltildi)
                // Robotlar çarpışmasın diye Y eksenindeki (sağ/sol) mesafeyi açtık
                if (leader_data.formation_type == FORMATION_SQUARE) {
                    if (my_id == 1) { dx = -0.15; dy =  0.30; } // SOL
                    if (my_id == 2) { dx = -0.15; dy = -0.30; } // SAĞ (Negatif)
                    if (my_id == 3) { dx = -0.40; dy =  0.00; } // ARKA
                } else {
                    if (my_id == 1) { dx = -0.25; dy =  0.20; }
                    if (my_id == 2) { dx = -0.25; dy = -0.20; }
                    if (my_id == 3) { dx = -0.50; dy =  0.00; }
                }

                // Global Hedef Hesaplama
                double c = cos(leader_data.theta);
                double s = sin(leader_data.theta);

                double target_x = leader_data.x + (dx * c - dy * s);
                double target_z = leader_data.z + (dx * s + dy * c);

                double error_x = target_x - my_pos[0];
                double error_z = target_z - my_pos[1];
                
                double dist = sqrt(error_x*error_x + error_z*error_z);
                double target_angle = atan2(error_z, error_x);
                
                double angle_diff = target_angle - my_pos[2];
                while(angle_diff > M_PI) angle_diff -= 2*M_PI;
                while(angle_diff < -M_PI) angle_diff += 2*M_PI;

                double base_v = 0;
                double rot_v = 0;

                // --- GELİŞMİŞ HAREKET KONTROLÜ ---
                
                // 1. Ölü Bölge (Deadband): Açı farkı çok azsa (0.1 radyan) titreşimi kes, düz git
                if (fabs(angle_diff) < 0.1) {
                    rot_v = 0;
                    base_v = dist * 10.0;
                }
                // 2. Büyük Açı Farkı: Olduğu yerde yavaşça dön
                else if (fabs(angle_diff) > 0.5) { 
                    base_v = 0.0; 
                    // Dönüş hızını azalttık ki hedefi kaçırıp sürekli dönmesin
                    rot_v = (angle_diff > 0) ? 1.5 : -1.5; 
                } 
                // 3. Normal Sürüş: Hem dön hem git
                else {
                    base_v = dist * 8.0; 
                    rot_v = angle_diff * 4.0;
                }
                
                if(base_v > MAX_SPEED) base_v = MAX_SPEED;

                left_speed = base_v - rot_v;
                right_speed = base_v + rot_v;
                
                // Engel varsa kaç
                if(obstacle) {
                    left_speed = b_left + base_v * 0.2;
                    right_speed = b_right + base_v * 0.2;
                }
            }
        }

        // Hız Limitleme
        if (left_speed > MAX_SPEED) left_speed = MAX_SPEED;
        if (left_speed < -MAX_SPEED) left_speed = -MAX_SPEED;
        if (right_speed > MAX_SPEED) right_speed = MAX_SPEED;
        if (right_speed < -MAX_SPEED) right_speed = -MAX_SPEED;

        if(left_motor) wb_motor_set_velocity(left_motor, left_speed);
        if(right_motor) wb_motor_set_velocity(right_motor, right_speed);
    }

    wb_robot_cleanup();
    return 0;
}
