#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/gps.h>
#include <webots/emitter.h>
#include <webots/receiver.h>

#define TIME_STEP 64
#define MAX_SPEED 3.0
#define COMMUNICATION_CHANNEL 1

// ps0 ile takipçiler arası mesafe (metre)
#define FOLLOW_DIST 0.22

// Takipçi kontrol
#define K_FWD  0.5  // mesafe -> ileri hız
#define K_TURN 0.1    // hedef sağ/sol -> dönüş gücü
#define STOP_DIST 0.03
#define ANGLE_DEAD 0.12

static double clamp(double v, double lo, double hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static double wrap_pi(double a) {
  while (a > M_PI) a -= 2.0*M_PI;
  while (a < -M_PI) a += 2.0*M_PI;
  return a;
}

static int get_robot_id() {
  const char *name = wb_robot_get_name();
  int n = (int)strlen(name);
  if (n > 0 && name[n-1] >= '0' && name[n-1] <= '9') return name[n-1] - '0';
  return 0;
}

// Lider paketi: sadece konum + yön (yaw) yolluyoruz.
// Yönü: GPS hız vektöründen (vx,vz) bulup hold ediyoruz.
typedef struct {
  double x, z;
  double theta;
  double vff;
} LeaderPacket;

int main() {
  wb_robot_init();

  WbDeviceTag left_motor  = wb_robot_get_device("left wheel motor");
  WbDeviceTag right_motor = wb_robot_get_device("right wheel motor");
  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);

  WbDeviceTag gps = wb_robot_get_device("gps");
  wb_gps_enable(gps, TIME_STEP);

  WbDeviceTag emitter  = wb_robot_get_device("emitter");
  WbDeviceTag receiver = wb_robot_get_device("receiver");
  if (emitter)  wb_emitter_set_channel(emitter, COMMUNICATION_CHANNEL);
  if (receiver) { wb_receiver_enable(receiver, TIME_STEP); wb_receiver_set_channel(receiver, COMMUNICATION_CHANNEL); }

  int my_id = get_robot_id();
  printf("RUNNING: %s (id=%d)\n", wb_robot_get_name(), my_id);

  // GPS türevleriyle lider heading
  bool have_last = false;
  double last_x = 0.0, last_z = 0.0;

  // Lider heading hold (hız çok küçükken zıplamasın)
  double theta_hold = 0.0;

  // follower packet
  LeaderPacket leader = {0};
  bool leader_ok = false;

  while (wb_robot_step(TIME_STEP) != -1) {
    if (wb_robot_get_time() < 1.0) {
      wb_motor_set_velocity(left_motor, 0.0);
      wb_motor_set_velocity(right_motor, 0.0);
      continue;
    }

    const double *p = wb_gps_get_values(gps);
    double x = p[0];
    double z = p[2];

    // velocity estimate
    double vx = 0.0, vz = 0.0;
    if (!have_last) {
      have_last = true;
      last_x = x; last_z = z;
    } else {
      double dt = TIME_STEP / 1000.0;
      vx = (x - last_x) / dt;
      vz = (z - last_z) / dt;
      last_x = x; last_z = z;
    }

    double left_speed = 0.0, right_speed = 0.0;

    // ============ LEADER ============
    if (my_id == 0) {
      double base = MAX_SPEED * 0.45;
      left_speed = base;
      right_speed = base;

      // leader heading (vx,vz): hız küçükse theta_hold
      double sp = sqrt(vx*vx + vz*vz);
      double theta = theta_hold;
      if (sp > 0.02) {
        theta = atan2(vz, vx);
        theta_hold = theta;
      }

      if (emitter) {
        LeaderPacket pkt;
        pkt.x = x; pkt.z = z;
        pkt.theta = theta;
        pkt.vff = 0.5 * (left_speed + right_speed);
        wb_emitter_send(emitter, &pkt, sizeof(pkt));
      }
    }

    // ============ FOLLOWERS ============
    if (my_id != 0) {
      if (receiver) {
        while (wb_receiver_get_queue_length(receiver) > 0) {
          if (wb_receiver_get_data_size(receiver) == (int)sizeof(LeaderPacket)) {
            memcpy(&leader, wb_receiver_get_data(receiver), sizeof(LeaderPacket));
            leader_ok = true;
          }
          wb_receiver_next_packet(receiver);
        }
      }

      if (leader_ok) {
        // Her takipçi liderin arkasında aynı çizgide ama farklı mesafede dursun:
        // ps1 en yakın, ps2 biraz daha, ps3 en arkada
        double behind = FOLLOW_DIST * (double)my_id; // id=1,2,3 => 0.22,0.44,0.66

        // hedef nokta: liderin arkasında (leader.theta doğrultusunda geri)
        double target_x = leader.x - behind * cos(leader.theta);
        double target_z = leader.z - behind * sin(leader.theta);

        double ex = target_x - x;
        double ez = target_z - z;
        double dist = sqrt(ex*ex + ez*ez);

        double target_angle = atan2(ez, ex);

        // *** DAİRE ÇİZMEYİ BİTİREN NUMARA ***
        // Takipçi kendi heading'ini hiç hesaplamıyor.
        // Dönüşü "hedef açı - liderin yönü" üzerinden yapıyoruz:
        // Böylece herkes liderin doğrultusuna kilitleniyor.
        double angle_diff = wrap_pi(target_angle - leader.theta);

        // İleri hız: liderle ak + gerideysen hızlan
        double base_v = leader.vff + K_FWD * dist;
        if (dist < STOP_DIST) base_v = leader.vff;
        base_v = clamp(base_v, 0.0, MAX_SPEED);

        // Dönüş: küçükse düz, büyükse biraz dön
        double rot = 0.0;
        if (fabs(angle_diff) > ANGLE_DEAD) rot = angle_diff * K_TURN;

        left_speed  = base_v - rot;
        right_speed = base_v + rot;
      }
    }

    left_speed  = clamp(left_speed,  -MAX_SPEED, MAX_SPEED);
    right_speed = clamp(right_speed, -MAX_SPEED, MAX_SPEED);

    wb_motor_set_velocity(left_motor, left_speed);
    wb_motor_set_velocity(right_motor, right_speed);
  }

  wb_robot_cleanup();
  return 0;
}
