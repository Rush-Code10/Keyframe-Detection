#!/usr/bin/env python3
"""
Simple Fan Control - Just turn fans on high during processing
No temperature monitoring, just raw fan control when needed.
"""

import os
import sys
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class SimpleFanControl:
    """Simple fan control - just turn fans on/off for processing."""
    
    def __init__(self):
        self.fans_active = False
        self.original_power_plan = None
        
    def start_fans(self):
        """Turn fans on high for processing."""
        if self.fans_active:
            return
            
        logger.info("🌀 Starting fan control for video processing")
        
        try:
            if sys.platform == "win32":
                self._start_windows_fans()
            elif sys.platform.startswith("linux"):
                self._start_linux_fans()
            else:
                logger.info("Fan control not implemented for this platform, but processing will continue")
            
            self.fans_active = True
            logger.info("✅ Fan control activated")
            
        except Exception as e:
            logger.warning(f"Could not activate fan control: {e}")
            logger.info("Processing will continue without fan control")
    
    def stop_fans(self):
        """Turn fans back to normal after processing."""
        if not self.fans_active:
            return
            
        logger.info("🛑 Stopping fan control, returning to normal")
        
        try:
            if sys.platform == "win32":
                self._stop_windows_fans()
            elif sys.platform.startswith("linux"):
                self._stop_linux_fans()
            
            self.fans_active = False
            logger.info("✅ Fan control deactivated, returned to normal")
            
        except Exception as e:
            logger.warning(f"Could not deactivate fan control: {e}")
            logger.info("You may need to manually restore power settings")
    
    def _start_windows_fans(self):
        """Windows fan control - set to high performance power plan."""
        try:
            # Get current power plan
            result = subprocess.run([
                "powershell", "-Command", 
                "powercfg /getactivescheme"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Extract current plan GUID
                output = result.stdout.strip()
                if "GUID:" in output:
                    current_guid = output.split("GUID:")[1].split()[0]
                    self.original_power_plan = current_guid
                    logger.info(f"Saved current power plan: {current_guid}")
            
            # Set to High Performance plan
            high_perf_guid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
            subprocess.run([
                "powershell", "-Command", 
                f"powercfg /setactive {high_perf_guid}"
            ], timeout=10, check=True)
            
            logger.info("Set power plan to High Performance for maximum cooling")
            
            # Also try to set CPU to maximum performance
            try:
                subprocess.run([
                    "powershell", "-Command",
                    "powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100"
                ], timeout=10)
                subprocess.run([
                    "powershell", "-Command",
                    "powercfg /setactive SCHEME_CURRENT"
                ], timeout=10)
                logger.info("Set CPU to maximum performance")
            except:
                pass  # Not critical if this fails
                
        except Exception as e:
            logger.debug(f"Windows fan control failed: {e}")
            raise
    
    def _stop_windows_fans(self):
        """Windows fan control - restore original power plan."""
        try:
            if self.original_power_plan:
                # Restore original power plan
                subprocess.run([
                    "powershell", "-Command", 
                    f"powercfg /setactive {self.original_power_plan}"
                ], timeout=10, check=True)
                logger.info(f"Restored original power plan: {self.original_power_plan}")
            else:
                # Default to Balanced plan
                balanced_guid = "381b4222-f694-41f0-9685-ff5bb260df2e"
                subprocess.run([
                    "powershell", "-Command", 
                    f"powercfg /setactive {balanced_guid}"
                ], timeout=10, check=True)
                logger.info("Set power plan back to Balanced")
                
        except Exception as e:
            logger.debug(f"Windows fan restore failed: {e}")
            raise
    
    def _start_linux_fans(self):
        """Linux fan control - set CPU governor to performance."""
        try:
            # Set CPU governor to performance mode
            subprocess.run([
                "sudo", "cpupower", "frequency-set", "-g", "performance"
            ], timeout=10, check=True)
            logger.info("Set CPU governor to performance mode")
            
            # Try to start fancontrol if available
            try:
                subprocess.run([
                    "sudo", "systemctl", "start", "fancontrol"
                ], timeout=10)
                logger.info("Started fancontrol service")
            except:
                pass  # Not critical
                
        except Exception as e:
            logger.debug(f"Linux fan control failed: {e}")
            raise
    
    def _stop_linux_fans(self):
        """Linux fan control - restore CPU governor."""
        try:
            # Set CPU governor back to powersave/ondemand
            for governor in ["ondemand", "powersave", "conservative"]:
                try:
                    subprocess.run([
                        "sudo", "cpupower", "frequency-set", "-g", governor
                    ], timeout=10, check=True)
                    logger.info(f"Set CPU governor back to {governor}")
                    break
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f"Linux fan restore failed: {e}")
            raise

# Global fan controller instance
_fan_controller = None

def start_processing_fans():
    """Start fans for video processing - call this when processing begins."""
    global _fan_controller
    if _fan_controller is None:
        _fan_controller = SimpleFanControl()
    
    _fan_controller.start_fans()

def stop_processing_fans():
    """Stop fans after video processing - call this when processing ends."""
    global _fan_controller
    if _fan_controller:
        _fan_controller.stop_fans()

def is_fan_control_active() -> bool:
    """Check if fan control is currently active."""
    global _fan_controller
    return _fan_controller.fans_active if _fan_controller else False