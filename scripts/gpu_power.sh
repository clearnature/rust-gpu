#!/bin/bash
# GPU 功率模式切换 (RX 9060 XT / GFX1200)
# 用法: sudo ./gpu_power.sh [quiet|auto|max]

CARD=/sys/class/drm/card1/device
SCLK=$CARD/pp_dpm_sclk
MCLK=$CARD/pp_dpm_mclk
LEVEL=$CARD/power_dpm_force_performance_level

status() {
    sclk=$(grep '\*' $SCLK | awk '{print $2}')
    mclk=$(grep '\*' $MCLK | awk '{print $2}')
    pwr=$(cat $CARD/hwmon/hwmon*/power1_average 2>/dev/null)
    temp=$(cat $CARD/hwmon/hwmon*/temp1_input 2>/dev/null)
    level=$(cat $LEVEL)
    echo "SCLK=$sclk MCLK=$mclk Power=$((${pwr:-0}/1000000))W Temp=$((${temp:-0}/1000))°C Level=$level"
}

case "${1:-status}" in
  quiet|q)
    echo manual > $LEVEL
    echo "0 0" > $SCLK
    echo "0" > $MCLK
    echo -n "[quiet] "; status
    ;;
  auto|a)
    echo auto > $LEVEL
    echo -n "[auto] "; status
    ;;
  max|m)
    echo manual > $LEVEL
    echo "1 1" > $SCLK
    echo "5" > $MCLK
    echo -n "[max] "; status
    ;;
  status|s)
    status
    ;;
  *)
    echo "用法: sudo $0 [quiet|auto|max|status]"
    echo "  quiet  - 静音省电"
    echo "  auto   - 自动调频 (推荐日常使用)"
    echo "  max    - 峰值性能 (训练用)"
    echo "  status - 查看当前状态"
    ;;
esac
