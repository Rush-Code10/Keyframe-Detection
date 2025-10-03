#!/usr/bin/env python3
"""
Fan Control Module for Thermal Management
Manages laptop fan speed during intensive video processing to prevent overheating.
"""

import os
import sys
import time
import logging
import threading
import subprocess
import psutil
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ThermalStatus:
    """Current thermal status of the system."""
    cpu_temp: float
    fan_speed: int
    cpu_usage: float
    memory_usage: float
    
class FanController:
    """Controls laptop fan speed for thermal management during processing."""
    
    def __init__(self, target_temp: float = 75.0, max_temp: float = 85.0):
        """
        Initialize fan controller.
        
        Args:
            target_temp: Target CPU temperature in Celsius
            max_temp: Maximum safe CPU temperature in Celsius
        """
        self.target_temp = target_temp
        self.max_temp = max_temp
        self.is_monitoring = False
        self.monitoring_thread = None
        self.original_fan_mode = None
        self.processing_active = False
        
        # Detection flags for different fan control methods
        self.has_speedfan = self._check_speedfan()
        self.has_nbfc = self._check_nbfc()
        self.has_fancontrol = self._check_fancontrol()
        self.has_powershell_thermal = self._check_powershell_thermal()
        
        logger.info(f"Fan controller initialized - Target: {target_temp}°C, Max: {max_temp}°C")
        
    def _check_speedfan(self) -> bool:
        """Check if SpeedFan is available (Windows)."""
        try:
            if sys.platform == "win32":
                # Check if SpeedFan is installed
                speedfan_paths = [
                    r"C:\Program Files\SpeedFan\speedfan.exe",
                    r"C:\Program Files (x86)\SpeedFan\speedfan.exe"
                ]
                return any(os.path.exists(path) for path in speedfan_paths)
        except Exception:
            pass
        return False
    
    def _check_nbfc(self) -> bool:
        """Check if NoteBook FanControl (NBFC) is available (Windows)."""
        try:
            if sys.platform == "win32":
                result = subprocess.run(["nbfc", "status"], capture_output=True, text=True, timeout=5)
                return result.returncode == 0
        except Exception:
            pass
        return False
    
    def _check_fancontrol(self) -> bool:
        """Check if fancontrol is available (Linux)."""
        try:
            if sys.platform.startswith("linux"):
                result = subprocess.run(["which", "fancontrol"], capture_output=True, timeout=5)
                return result.returncode == 0
        except Exception:
            pass
        return False
    
    def _check_powershell_thermal(self) -> bool:
        """Check if PowerShell thermal management is available (Windows)."""
        try:
            if sys.platform == "win32":
                # Try multiple PowerShell thermal methods
                methods = [
                    "Get-WmiObject -Namespace root/wmi -Class MSAcpi_ThermalZoneTemperature",
                    "Get-WmiObject -Namespace root/OpenHardwareMonitor -Class Sensor",
                    "Get-WmiObject -Namespace root/LibreHardwareMonitor -Class Sensor",
                    "Get-Counter '\\Thermal Zone Information(*)\\Temperature' -ErrorAction SilentlyContinue"
                ]
                
                for method in methods:
                    try:
                        cmd = ["powershell", "-Command", method]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                        if result.returncode == 0 and result.stdout.strip():
                            logger.debug(f"PowerShell thermal method available: {method}")
                            return True
                    except Exception:
                        continue
        except Exception:
            pass
        return False
    
    def get_cpu_temperature(self) -> Optional[float]:
        """Get current CPU temperature in Celsius."""
        try:
            # Try psutil first (works on many systems)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Look for CPU-specific sensors first
                    for name, entries in temps.items():
                        name_lower = name.lower()
                        if any(keyword in name_lower for keyword in ['cpu', 'core', 'processor']):
                            for entry in entries:
                                if hasattr(entry, 'current') and entry.current:
                                    temp = entry.current
                                    if 20 <= temp <= 120:  # Reasonable range
                                        logger.debug(f"psutil CPU temperature: {temp}°C from {name}")
                                        return temp
                    
                    # If no CPU specific, try any temperature sensor
                    for name, entries in temps.items():
                        for entry in entries:
                            if hasattr(entry, 'current') and entry.current:
                                temp = entry.current
                                if 20 <= temp <= 120:
                                    logger.debug(f"psutil general temperature: {temp}°C from {name}")
                                    return temp
            
            # Platform-specific methods
            if sys.platform == "win32":
                temp = self._get_windows_cpu_temp()
                if temp:
                    return temp
            elif sys.platform.startswith("linux"):
                temp = self._get_linux_cpu_temp()
                if temp:
                    return temp
            
            # Fallback: Estimate temperature based on CPU usage (rough approximation)
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 0:
                # Rough estimation: idle temp around 40°C, high usage can reach 80°C+
                estimated_temp = 40 + (cpu_usage * 0.5)  # Very rough approximation
                logger.debug(f"Using estimated temperature based on CPU usage: {estimated_temp:.1f}°C (CPU: {cpu_usage}%)")
                return estimated_temp
                
        except Exception as e:
            logger.debug(f"Could not get CPU temperature: {e}")
        
        return None
    
    def _get_windows_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature on Windows using multiple methods."""
        # Method 1: Try Open Hardware Monitor via WMI
        try:
            cmd = ["powershell", "-Command", 
                  "Get-WmiObject -Namespace root/OpenHardwareMonitor -Class Sensor | Where-Object {$_.SensorType -eq 'Temperature' -and $_.Name -like '*CPU*'} | Select-Object -First 1 -ExpandProperty Value"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                temp = float(result.stdout.strip())
                if 20 <= temp <= 120:  # Reasonable temperature range
                    logger.debug(f"OpenHardware temperature: {temp}°C")
                    return temp
        except Exception as e:
            logger.debug(f"OpenHardware method failed: {e}")
        
        # Method 2: Try WMI thermal zones
        try:
            cmd = ["powershell", "-Command", 
                  "Get-WmiObject -Namespace root/wmi -Class MSAcpi_ThermalZoneTemperature | ForEach-Object {($_.CurrentTemperature - 2732) / 10.0}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    try:
                        temp = float(line.strip())
                        if 20 <= temp <= 120:  # Reasonable temperature range
                            logger.debug(f"WMI thermal zone temperature: {temp}°C")
                            return temp
                    except ValueError:
                        continue
        except Exception as e:
            logger.debug(f"WMI thermal zone method failed: {e}")
        
        # Method 3: Try LibreHardwareMonitor WMI
        try:
            cmd = ["powershell", "-Command", 
                  "Get-WmiObject -Namespace root/LibreHardwareMonitor -Class Sensor | Where-Object {$_.SensorType -eq 'Temperature' -and $_.Name -like '*CPU*'} | Select-Object -First 1 -ExpandProperty Value"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                temp = float(result.stdout.strip())
                if 20 <= temp <= 120:
                    logger.debug(f"LibreHardware temperature: {temp}°C")
                    return temp
        except Exception as e:
            logger.debug(f"LibreHardware method failed: {e}")
        
        # Method 4: Try Windows Performance Toolkit (if available)
        try:
            cmd = ["powershell", "-Command", 
                  "Get-Counter '\\Thermal Zone Information(*)\\Temperature' -ErrorAction SilentlyContinue | ForEach-Object {$_.CounterSamples} | ForEach-Object {($_.CookedValue - 273.15)}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    try:
                        temp = float(line.strip())
                        if 20 <= temp <= 120:
                            logger.debug(f"Performance counter temperature: {temp}°C")
                            return temp
                    except ValueError:
                        continue
        except Exception as e:
            logger.debug(f"Performance counter method failed: {e}")
        
        # Method 5: Try CoreTemp if available
        try:
            import winreg
            # Check if Core Temp is installed and running
            key_path = r"SOFTWARE\CoreTemp"
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                    temp_count = winreg.QueryValueEx(key, "TjMax")[0]
                    if temp_count > 0:
                        # Try to read first CPU core temperature
                        temp = winreg.QueryValueEx(key, "Temp0")[0]
                        if 20 <= temp <= 120:
                            logger.debug(f"CoreTemp temperature: {temp}°C")
                            return temp
            except FileNotFoundError:
                pass
        except Exception as e:
            logger.debug(f"CoreTemp method failed: {e}")
        
        logger.debug("All Windows temperature methods failed")
        return None
    
    def _get_linux_cpu_temp(self) -> Optional[float]:
        """Get CPU temperature on Linux."""
        try:
            # Try thermal zone files
            thermal_paths = [
                "/sys/class/thermal/thermal_zone0/temp",
                "/sys/class/thermal/thermal_zone1/temp"
            ]
            
            for path in thermal_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp_millicelsius = int(f.read().strip())
                        return temp_millicelsius / 1000.0
                        
        except Exception as e:
            logger.debug(f"Linux temperature check failed: {e}")
        
        return None
    
    def set_fan_performance_mode(self, enable: bool = True):
        """Set fan to performance mode for intensive processing."""
        try:
            if sys.platform == "win32":
                self._set_windows_fan_mode(enable)
            elif sys.platform.startswith("linux"):
                self._set_linux_fan_mode(enable)
            else:
                logger.warning("Fan control not supported on this platform")
                
        except Exception as e:
            logger.error(f"Failed to set fan mode: {e}")
    
    def _set_windows_fan_mode(self, enable: bool):
        """Set fan mode on Windows."""
        try:
            if self.has_nbfc:
                # Use NBFC if available
                if enable:
                    subprocess.run(["nbfc", "set", "-a"], timeout=10)
                    logger.info("Set fan to automatic mode via NBFC")
                else:
                    subprocess.run(["nbfc", "set", "-s", "0"], timeout=10)
            else:
                # Use PowerShell to set power profile for better cooling
                if enable:
                    # Set to High Performance power plan for better cooling
                    cmd = ["powershell", "-Command", 
                          "powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"]
                    subprocess.run(cmd, timeout=10)
                    logger.info("Set power plan to High Performance for better cooling")
                else:
                    # Set back to Balanced
                    cmd = ["powershell", "-Command", 
                          "powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e"]
                    subprocess.run(cmd, timeout=10)
                    logger.info("Set power plan back to Balanced")
                    
        except Exception as e:
            logger.warning(f"Windows fan control failed: {e}")
    
    def _set_linux_fan_mode(self, enable: bool):
        """Set fan mode on Linux."""
        try:
            if self.has_fancontrol:
                if enable:
                    subprocess.run(["sudo", "systemctl", "start", "fancontrol"], timeout=10)
                    logger.info("Started fancontrol service")
                else:
                    subprocess.run(["sudo", "systemctl", "stop", "fancontrol"], timeout=10)
            else:
                # Try to set CPU governor for better thermal management
                try:
                    if enable:
                        subprocess.run(["sudo", "cpupower", "frequency-set", "-g", "performance"], timeout=10)
                        logger.info("Set CPU governor to performance mode")
                    else:
                        subprocess.run(["sudo", "cpupower", "frequency-set", "-g", "powersave"], timeout=10)
                        logger.info("Set CPU governor back to powersave")
                except Exception:
                    logger.debug("CPU governor control not available")
                    
        except Exception as e:
            logger.warning(f"Linux fan control failed: {e}")
    
    def get_thermal_status(self) -> ThermalStatus:
        """Get current thermal status."""
        cpu_temp = self.get_cpu_temperature() or 0.0
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Try to get fan speed (difficult without specialized tools)
        fan_speed = self._get_fan_speed()
        
        return ThermalStatus(
            cpu_temp=cpu_temp,
            fan_speed=fan_speed,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent
        )
    
    def _get_fan_speed(self) -> int:
        """Get current fan speed (RPM or percentage)."""
        try:
            if hasattr(psutil, "sensors_fans"):
                fans = psutil.sensors_fans()
                if fans:
                    for name, entries in fans.items():
                        if entries:
                            return entries[0].current
        except Exception:
            pass
        return 0  # Unknown
    
    def start_monitoring(self, processing_callback=None):
        """Start thermal monitoring during processing."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.processing_active = True
        
        # Set fan to performance mode
        self.set_fan_performance_mode(True)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_thermal, 
                                                 args=(processing_callback,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Started thermal monitoring and fan control")
    
    def stop_monitoring(self):
        """Stop thermal monitoring and restore normal fan operation."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.processing_active = False
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Restore normal fan mode
        self.set_fan_performance_mode(False)
        
        logger.info("Stopped thermal monitoring and restored normal fan operation")
    
    def _monitor_thermal(self, processing_callback=None):
        """Monitor thermal status and adjust fan accordingly."""
        logger.info("Thermal monitoring started")
        
        overheated_count = 0
        last_temp_log = 0
        
        while self.is_monitoring and self.processing_active:
            try:
                status = self.get_thermal_status()
                current_time = time.time()
                
                # Log temperature every 30 seconds
                if current_time - last_temp_log > 30:
                    logger.info(f"Thermal status - CPU: {status.cpu_temp:.1f}°C, "
                              f"Usage: {status.cpu_usage:.1f}%, Memory: {status.memory_usage:.1f}%")
                    last_temp_log = current_time
                
                # Check for overheating
                if status.cpu_temp > self.max_temp:
                    overheated_count += 1
                    logger.warning(f"CPU temperature high: {status.cpu_temp:.1f}°C (max: {self.max_temp}°C)")
                    
                    if overheated_count >= 3:  # 3 consecutive high readings
                        logger.error("System overheating detected! Consider pausing processing.")
                        if processing_callback:
                            processing_callback("thermal_warning", status)
                else:
                    overheated_count = 0
                
                # Sleep for monitoring interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in thermal monitoring: {e}")
                time.sleep(30)  # Wait longer on error
        
        logger.info("Thermal monitoring stopped")

class ThermalManager:
    """High-level thermal management for video processing."""
    
    def __init__(self):
        self.fan_controller = FanController()
        self.is_processing = False
    
    def start_processing_thermal_management(self, processing_callback=None):
        """Start thermal management for intensive processing."""
        if self.is_processing:
            return
        
        self.is_processing = True
        logger.info("🌡️ Starting thermal management for video processing")
        
        # Start fan control and monitoring
        self.fan_controller.start_monitoring(processing_callback)
        
        # Log system info
        status = self.fan_controller.get_thermal_status()
        logger.info(f"Initial thermal status - CPU: {status.cpu_temp:.1f}°C, "
                   f"Usage: {status.cpu_usage:.1f}%")
    
    def stop_processing_thermal_management(self):
        """Stop thermal management and restore normal operation."""
        if not self.is_processing:
            return
        
        self.is_processing = False
        logger.info("🌡️ Stopping thermal management")
        
        # Stop fan control
        self.fan_controller.stop_monitoring()
        
        # Log final status
        status = self.fan_controller.get_thermal_status()
        logger.info(f"Final thermal status - CPU: {status.cpu_temp:.1f}°C")
    
    def get_thermal_info(self) -> Dict[str, Any]:
        """Get current thermal information."""
        status = self.fan_controller.get_thermal_status()
        
        return {
            'cpu_temperature': status.cpu_temp,
            'cpu_usage': status.cpu_usage,
            'memory_usage': status.memory_usage,
            'fan_speed': status.fan_speed,
            'thermal_management_active': self.is_processing,
            'supports_fan_control': (self.fan_controller.has_nbfc or 
                                   self.fan_controller.has_fancontrol or
                                   self.fan_controller.has_powershell_thermal)
        }