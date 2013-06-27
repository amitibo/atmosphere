"""
Clean all temp files from all nodes.
"""

from __future__ import division
import shutil
import glob
import os
import pwd
import paramiko


def main():
    """Main doc """
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    for i in range(1, 109):
        try:
            ssh.connect('n%s' % ('000%d' % i)[-3:])
        except Exception, e:
            print e
            continue
        
        stdin, stdout, stderr = ssh.exec_command('hostname')
        print stdout.readlines()[0].strip()
    
        stdin, stdout, stderr = ssh.exec_command('ls -l /gtmp/')
        
        for line in stdout.readlines():
            line = line.strip()
                        
            try:
                parts = line.split()
                owner = parts[2]
                path = parts[-1]
            except:
                continue

            if owner == 'amitibo':
                try:
                    sin, sout, serr = ssh.exec_command('rm -r -f %s' % path)
                except Exception, e:
                    print e
                
        ssh.close()
        

if __name__ == '__main__':
    main()

    
    
