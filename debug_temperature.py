#!/usr/bin/env python3
"""
Test script specifically for temperature detection on Windows
"""

import sys
import os
import subprocess
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_temperature_methods():
    """Test various temperature detection methods"""
    print("🌡️ Temperature Detection Test")
    print("=" * 50)
    
    # Test 1: psutil sensors
    print("\n1. Testing psutil sensors...")
    try:
        import psutil
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                print(f"   Found {len(temps)} sensor groups:")
                for name, entries in temps.items():
                    print(f"   • {name}: {len(entries)} sensors")
                    for i, entry in enumerate(entries):
                        if hasattr(entry, 'current'):
                            print(f"     - Sensor {i}: {entry.current}°C (label: {getattr(entry, 'label', 'N/A')})")
            else:
                print("   ❌ No temperature sensors found via psutil")
        else:
            print("   ❌ psutil.sensors_temperatures not available")
    except Exception as e:
        print(f"   ❌ psutil test failed: {e}")
    
    # Test 2: Windows WMI thermal zones
    print("\n2. Testing Windows WMI thermal zones...")
    try:
        cmd = ["powershell", "-Command", 
              "Get-WmiObject -Namespace root/wmi -Class MSAcpi_ThermalZoneTemperature | ForEach-Object {Write-Host \"Zone: $($_.InstanceName), Temp: $(($_.CurrentTemperature - 2732) / 10.0)°C\"}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            print("   ✅ WMI thermal zones found:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line.strip()}")
        else:
            print("   ❌ No WMI thermal zones found")
            if result.stderr:
                print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"   ❌ WMI test failed: {e}")
    
    # Test 3: Open Hardware Monitor
    print("\n3. Testing Open Hardware Monitor...")
    try:
        cmd = ["powershell", "-Command", 
              "Get-WmiObject -Namespace root/OpenHardwareMonitor -Class Sensor | Where-Object {$_.SensorType -eq 'Temperature'} | ForEach-Object {Write-Host \"$($_.Name): $($_.Value)°C\"}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            print("   ✅ Open Hardware Monitor sensors found:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line.strip()}")
        else:
            print("   ❌ Open Hardware Monitor not available")
    except Exception as e:
        print(f"   ❌ Open Hardware Monitor test failed: {e}")
    
    # Test 4: LibreHardwareMonitor
    print("\n4. Testing LibreHardwareMonitor...")
    try:
        cmd = ["powershell", "-Command", 
              "Get-WmiObject -Namespace root/LibreHardwareMonitor -Class Sensor | Where-Object {$_.SensorType -eq 'Temperature'} | ForEach-Object {Write-Host \"$($_.Name): $($_.Value)°C\"}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            print("   ✅ LibreHardwareMonitor sensors found:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line.strip()}")
        else:
            print("   ❌ LibreHardwareMonitor not available")
    except Exception as e:
        print(f"   ❌ LibreHardwareMonitor test failed: {e}")
    
    # Test 5: Performance Counters
    print("\n5. Testing Performance Counters...")
    try:
        cmd = ["powershell", "-Command", 
              "Get-Counter '\\Thermal Zone Information(*)\\Temperature' -ErrorAction SilentlyContinue | ForEach-Object {$_.CounterSamples} | ForEach-Object {Write-Host \"$($_.Path): $(($_.CookedValue - 273.15).ToString('F1'))°C\"}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            print("   ✅ Performance counter thermal data found:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line.strip()}")
        else:
            print("   ❌ No performance counter thermal data")
    except Exception as e:
        print(f"   ❌ Performance counter test failed: {e}")
    
    # Test 6: Our improved thermal manager
    print("\n6. Testing improved thermal manager...")
    try:
        from modules.thermal_manager import FanController
        controller = FanController()
        temp = controller.get_cpu_temperature()
        if temp and temp > 0:
            print(f"   ✅ Thermal manager detected temperature: {temp:.1f}°C")
        else:
            print("   ❌ Thermal manager could not detect temperature")
    except Exception as e:
        print(f"   ❌ Thermal manager test failed: {e}")
    
    # Test 7: CPU usage based estimation
    print("\n7. Testing CPU usage estimation...")
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=2)
        estimated_temp = 40 + (cpu_usage * 0.5)
        print(f"   CPU Usage: {cpu_usage:.1f}%")
        print(f"   Estimated Temperature: {estimated_temp:.1f}°C")
        print("   (This is a rough approximation when sensors aren't available)")
    except Exception as e:
        print(f"   ❌ CPU usage test failed: {e}")

def test_with_load():
    """Test temperature detection under CPU load"""
    print(f"\n🔥 Testing with CPU load (10 seconds)...")
    print("   Creating CPU load to see if temperature detection works...")
    
    import threading
    import time
    
    # Create CPU load
    stop_load = False
    def cpu_load():
        while not stop_load:
            pass  # Busy loop
    
    # Start load threads
    threads = []
    for i in range(4):  # 4 threads to load CPU
        t = threading.Thread(target=cpu_load)
        t.daemon = True
        t.start()
        threads.append(t)
    
    try:
        from modules.thermal_manager import FanController
        controller = FanController()
        
        for i in range(10):
            temp = controller.get_cpu_temperature()
            cpu_usage = __import__('psutil').cpu_percent(interval=0.1)
            print(f"   Load test {i+1}/10: Temp: {temp:.1f}°C, CPU: {cpu_usage:.1f}%")
            time.sleep(1)
    except Exception as e:
        print(f"   ❌ Load test failed: {e}")
    finally:
        stop_load = True
        time.sleep(0.5)  # Let threads finish

if __name__ == "__main__":
    print("Temperature Detection Diagnostic Tool")
    print("This will help identify why temperature stays at 0°C\n")
    
    test_temperature_methods()
    
    # Ask user if they want to run load test
    try:
        choice = input("\nRun CPU load test to see temperature changes? (y/n): ").lower()
        if choice == 'y':
            test_with_load()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    print("\n📊 Summary:")
    print("If all methods show 0°C or fail, your system may not expose")
    print("temperature sensors through standard Windows interfaces.")
    print("The system will use CPU usage estimation as a fallback.")