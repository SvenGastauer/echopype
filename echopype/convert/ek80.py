"""Read and Convert Simrad EK80 raw files


TO DO :
    Pulse compression
    Angular Position

    Impedance readings
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from lxml import etree
from itertools import groupby

from datetime import datetime, timedelta

from struct import unpack

from scipy.signal import chirp
from scipy.signal import resample

class convertEK80(object):
    def __init__(self, _fn=""):
        self.SAMPLE_REGEX =  b'NME0|XML0|RAW3|TAG0|self.mru0|FIL1'
        self.BLOCK_SIZE = 1024 * 40 #Block size for search radius
        self.LENGTH_SIZE = 4
        self.DATAGRAM_HEADER_SIZE = 12
        self.pings = defaultdict(list)
    
        self.NMEA = defaultdict(list)
        self.environment = defaultdict(list)
        self.parameters = defaultdict(list)
        self.filters = defaultdict(list)
        self.mru = defaultdict(list)
        #counters
        self.count_NMEA = -1
        self.count_env = -1
        self.count_para = -1
        self.count_filt = -1
        self.count_mru = -1
        
        self.fn = _fn

    def cmpPwrEK80(self, ping, config, ztrd = 75):
        """Compute power from complex signal
        
        This function assumes that a config was read previously and information such as impedance and number of transducers is available
        
        Parameters
        ----------
        ping: dict
            ping dictionary that will be updated
        ztrd: int
            Nominal Impedance, set to 75 Ohms by default

        """

        cid = ping['cid']
        impedance = int(config[cid]['Impedance'])
        y = ping['comp_sig']
        nb_elem = len(y)
        y = pd.DataFrame(y)
        y = y.sum(axis = 1) / nb_elem
        power = nb_elem * ( abs(y) / (2 * np.sqrt(2) )) **2 * (( int(impedance) + ztrd ) / int(impedance) )**2 / ztrd
        ping.update({'y':y})
        ping.update({'power':power})
         

    def xml2d(self, e):      
        """Convert an etree into a dict structure
        
        This function retruns the dictionary representation of the XML tree

        Parameters
        ----------
        e: etree.Element
            the root of the tree
        
        """
        
        def _xml2d(e):
            kids = dict(e.attrib)
            if e.text:
                kids['__text__'] = e.text
            if e.tail:
                kids['__tail__'] = e.tail
            for k, g in groupby(e, lambda x: x.tag):
                g = [ _xml2d(x) for x in g ] 
                kids[k]=  g
            return kids
        return { e.tag : _xml2d(e) }

    def parse_XML(self, xml_string):
        """parse xml into dict

        This function parses an XML string and returns a dictionary

        Parameters
        ----------
        xml_string: str
            an xml string
        """
        parser = etree.XMLParser(recover=True)
        xml_info = etree.fromstring(xml_string, parser=parser)
        
        return self.xml2d(xml_info)
    
    def pulse_generator(self, chID, ping):
        """Generate a simrad EK80 pulse
        
        Simulates a pulse, needed for matched filtering
        
        Parameters
        ----------
        
        chID: int
            The channel ID for which the pulse should be simulated
        ping: int
            The ping number for which the pulse should be simulated
            
        """
        f_s_ori=1.5*1e6;
        
        F1 = self.filters[chID+1*2-2]
        F2 = self.filters[chID+1*2-2+1]
        D_1 = F1['DecFac']
        D_2 = F2['DecFac']
        
        filt_1 = F1['Coeff'][0][0::2]+ 1j * F1['Coeff'][0][1::2]
        filt_2 = F2['Coeff'][0][0::2] + 1j * F2['Coeff'][0][1::2]
        
        
        para  = pd.DataFrame(self.parameters).transpose()
        para = para[para['ChannelID']== self.CID[chID]].iloc[ping]
        
        f_s_sig = 1 / float(para['SampleInterval'])
        FreqStart = int(para['FrequencyStart'])
        FreqEnd = int(para['FrequencyEnd'])
        
        pulse_length = float(para['PulseDuration'])
        pulse_slope = float(para['Slope'])
        
        t_sim_pulse = np.transpose(np.linspace(0,pulse_length,int(pulse_length * f_s_ori)))
        
        n_p = len(t_sim_pulse)
        
        nwtx = 2 * np.floor( pulse_slope * n_p)
        
        wtxtmp  = np.hanning(nwtx)
        
        nwtxh   = int(np.ceil(nwtx/2))
        
        
        env_pulse = np.concatenate([wtxtmp[0:nwtxh], np.ones(int(n_p - nwtx)), wtxtmp[nwtxh:]])
        
        sim_pulse = env_pulse * chirp(t = t_sim_pulse,
                                      f0 = FreqStart,
                                      f1=FreqEnd,
                                      t1=t_sim_pulse[-1])
        
        #Filter simulated pulse to create match filter%
        f_s_dec = []
        f_s_dec.append(f_s_ori / D_1)
        f_s_dec.append(f_s_dec[0] / D_2)
        
        if abs(f_s_dec[1] - f_s_sig) >= 1:
            print('Decimated pulse sample rate not matching signal sampling rate')
        #make convolution function with padding
        def conv(u,v):
            npad = len(v) - 1
            u_padded = np.pad(u, (npad // 2, npad - npad // 2), mode='constant')
            return(np.convolve(u_padded, v, 'valid'))
        
        sim_pulse_1 = conv(sim_pulse / max(abs(sim_pulse)), filt_1) 
        sim_pulse_1 = resample(sim_pulse_1,int(np.floor(len(sim_pulse)/D_1)))
        
        
        sim_pulse_2 = conv(sim_pulse_1 / max(abs(sim_pulse_1)),filt_2)
        sim_pulse_2 = resample(sim_pulse_2,int(np.floor(len(sim_pulse)/D_2)))
        
        sim_pulse_2 = sim_pulse_2 / max(sim_pulse_2)
        
        y_tx_matched = np.conj(np.flipud(sim_pulse_2))
        
        return(sim_pulse_2, y_tx_matched)

    def matched_filter(self,chID):
        """Create matched Filter
        
        Parameters
        ----------
        chID: int
            Channel ID (selected tranceiver)
        
        """
        if 'FrequencyStart' in self.params.iloc[chID]:
            if self.params.iloc[chID]['FrequencyStart'] is not self.params.iloc[chID]['FrequencyEnd']:
                mode = 'FM'
                comp_sig  = self.comp_sig_data[chID]
                
                nb_pings, nb_samples, nb_chan = np.shape(comp_sig)
                _, y_tx_matched = self.pulse_generator(chID,0)
                data = [[]] * nb_chan
                for ch in range(nb_chan):
                    cp = comp_sig[:,:,ch]
                    
                    val_sq=sum( abs(y_tx_matched) **2)
                    n = len(y_tx_matched) + nb_samples 
                    yc = np.fft.ifft(np.fft.fft(y_tx_matched,n) * np.fft.fft(cp,n,ch)/val_sq)
                    data[ch] = yc[:,len(y_tx_matched):]
                return data,mode
                    
            else:
                mode = 'CW'
                return mode
        else:
            mode = 'CW'
            return mode
        
    def _index_ek80(self):
        """create index of datagrams
        Creates an index of the location and length of the datagrams in the raw file
        The function returns a pandas dataframe with datagram type ('datagram'), start position ('start') and length of the datagram as columns

        Parameters
        ----------
        fn: str
            Filename and location as string
        
        """
        ind2 = []
        with open(self.fn ,'rb') as bin_file:
            #bin_file.seek(7)
            position = bin_file.tell()
            
            raw = bin_file.read(self.BLOCK_SIZE)
            while len(raw) > 4:
        
                for match in re.finditer(b'NME0|XML0|RAW3|TAG0|self.mru0|FIL1', raw):
                    
                    if match:
                        if match.span()[0] >= 4:
                           l =  unpack('i', raw[match.span()[0]-self.LENGTH_SIZE : match.span()[0]])[0]
                           ind2.append([match.group(), position + match.span()[0],l])
                    else:
                        bin_file.seek(position + self.BLOCK_SIZE - 4)
                position = bin_file.tell()
                # Read the next block for regex search
                bin_file.seek(position - 4)
                position = position - 4
                raw = bin_file.read(self.BLOCK_SIZE)
        idx = pd.DataFrame(ind2, columns=['datagram','start','length'])
        return(idx)
            
    def readEK80(self):
        """create index of datagrams
    
        Reads the EK80 raw file based on a file index and puts the information into the object
    
        """
    
        #get file index
        idx = self._index_ek80()

        with open(self.fn , 'rb') as file:
            file.seek(0)
            for i in range(len(idx)):
                #######
                # XML0
                #######
                if idx['datagram'][i] == b'XML0':
                    file.seek(idx['start'][i]+self.DATAGRAM_HEADER_SIZE)
                    xml_string = np.fromfile(file, dtype = 'a'+str(idx['length'][i] - self.DATAGRAM_HEADER_SIZE),count=1)[0]
                    xml0_dict = self.parse_XML(xml_string)
                    xmltype = list(xml0_dict.keys())[0]
                    #########
                    # Config
                    #########
                    if xmltype == 'Configuration':
                        xml_dat = xml0_dict['Configuration']
                        self.sensors = xml_dat['ConfiguredSensors'][0]
                        self.config_header = xml_dat['Header'][0]
                        self.config_transceiver = xml_dat['Transceivers'][0]['Transceiver']
                        self.config_transducer = xml_dat['Transducers'][0]['Transducer']
                        self.n_trans = len(self.config_transceiver)
                        self.CID =[]
                        for c in range(self.n_trans):
                            self.CID.append(xml_dat['Transceivers'][0]['Transceiver'][c]['Channels'][0]['Channel'][0]['ChannelID'])
                            self.pings[c] = defaultdict(list)
                        ping_count = [-1] * len(self.CID)
                    #########
                    # environment
                    #########
                    if xmltype == 'Environment':
                        self.count_env += 1
                        self.environment[self.count_env] = xml0_dict['Environment']
                    #########
                    # Ping self.parameters
                    #########
                    if xmltype == 'Parameter':
                        self.count_para += 1
                        self.parameters[self.count_para] = xml0_dict['Parameter']['Channel'][0]
                
                ###########
                # self.mru - Motion
                ###########
                elif idx['datagram'][i] == b'MRU0':
                    self.count_mru += 1
                    file.seek(idx['start'][i]+self.DATAGRAM_HEADER_SIZE)
                    mru_dtype = np.dtype([('heave','f4'),('roll','f4'),('pitch','f4'),('heading','f4')])
                    var = ['heave','roll','pitch','heading']
                    mru = np.fromfile(file, dtype=mru_dtype, count=1)
                    self.mru[self.count_mru] = dict(zip(var,mru[0]))
                ###########
                # FIL1 - Filter data
                ###########
                elif idx['datagram'][i] == b'FIL1':
                    self.count_filt += 1
                    file.seek(idx['start'][i]+self.DATAGRAM_HEADER_SIZE)
                    FIL1_dtype = np.dtype([('Stage','i2'),('Spare','i2'), ('Channel','S128'),('NCoeff','i2'),('DecFac','i2')])
                    f1 = np.fromfile(file, dtype = FIL1_dtype, count=1)
                    coeffs = np.fromfile(file,dtype = str(2 * f1['NCoeff'][0]) + 'f4', count=1)
                    self.filters[self.count_filt] = dict({'ChID':f1['Channel'][0].decode('ascii'),'Stage':f1['Stage'][0],'Spare':f1['Spare'][0],'NCoeff':f1['NCoeff'][0],'DecFac':f1['DecFac'][0], 'Coeff': coeffs})
                ################################################
                ##### NME0 Reader
                ################################################
                elif idx['datagram'][i] == b'NME0':
                    self.count_NMEA += 1
                    file.seek(idx['start'][i])
                    head = np.fromfile(file,dtype='3i4',count=1)
                    self.NMEA_string = np.fromfile(file,dtype='a'+str(idx['length'][i]),count=1)
                    if len(self.NMEA_string) > 0:
                        self.NMEA_string = self.NMEA_string[0]
                    if len(self.NMEA_string) > 6:
                        self.NMEA_type = self.NMEA_string[3:6] 
                        self.NMEA_ori = self.NMEA_string[1:3] #SD for sounder, Depth
                        self.NMEA[self.count_NMEA] = dict({'string': self.NMEA_string,'type':self.NMEA_type,'ori':self.NMEA_ori})
                    
                ########
                # RAW 3
                ########
                elif idx['datagram'][i] == b'RAW3':
                    file.seek(idx['start'][i]+4)
                    
                    #get ping header information into temporary ping dictionary
                    h_dtype = np.dtype([('lowDate','i4'),
                                        ('highDate','i4'),
                                        ('chID', 'S128'),
                                        ('dataType','I'),
                                        ('Offset','I'),
                                        ('SampleCount','I')])
                    head = np.fromfile(file, dtype = h_dtype, count=1)
                    head = dict(zip(['lowDate', 'highDate','chID', 'dataType','Offset','SampleCount'] ,list(head.tolist()[0])))
                    ntSecs = (head['highDate'] * 2 ** 32 + head['lowDate']) / 10000000
                    head.update({'dgTime' : datetime(1601, 1, 1, 0, 0, 0) + timedelta(seconds = ntSecs)})
                
                    #get channel ID
                    head.update({'cid' : self.CID.index(head['chID'].decode('ascii'))})
                    
                    #get current ping number
                    ping_count[head['cid']] += 1
                    ping_n = ping_count[head['cid']]
                    
                    #get sample count
                    sampleCount = head['SampleCount']
                     
                    #convert datatype to binary and reverse
                    dTbin = format(head['dataType'],'b')[::-1]
                    
                    #check file format for complex, and angular information to compute power
                    if dTbin[0:10][0] == format(1,'b'):
                        if dTbin[0:10][1] == format(1, 'b'):
                            #power and angular information
                            if sampleCount * 4 == idx['length'][i] - self.DATAGRAM_HEADER_SIZE - 12 - 128:
                                head.update({'power': np.fromfile(file, 
                                                      dtype = str(int(sampleCount)) + 'i2', 
                                                      count=1)[0] * 10. * np.log10(2) / 256.})
                                angles = np.fromfile(file, 
                                                      dtype = str(int(2*sampleCount)) + 'i1', 
                                                      count=1)[0].reshape(2,sampleCount)
                                head.update({'acrossPhi': angles[0]})
                                head.ip({'alongPhi': angles[1]})
                            else:
                                #only power information
                                 head.update({'power': np.fromfile(file, 
                                                                        dtype = str(int(sampleCount)) + 'i2', 
                                                                        count=1)[0] * 10. * np.log10(2) / 256. })
                    #...complex samples are available
                    else: 
                    
                        nb_cplx_per_samples =int(dTbin[7:][::-1],2)
                         
                        if dTbin[3] == format(1, 'b'):
                             fmt = 'f' #'float32'
                             ds = 4 #4 bytes
                        elif dTbin[2] == format(1, 'b'):
                             fmt = 'e' #'float16'
                             ds = 2 #2 bytes
                        if sampleCount > 0:
                            temp = np.fromfile(file, 
                                               dtype=np.dtype([('cplx_samples', fmt + str(ds))]), 
                                               count=int(nb_cplx_per_samples) * int(sampleCount))
                            temp =np.array([i[0] for i in temp]).T
                            temp = temp.reshape(int(sampleCount),-1)
                            if temp.size % nb_cplx_per_samples != 0:
                                sampleCount = 0
                            else:
                                sampleCount = temp.size / nb_cplx_per_samples
                            comp_sig = {}
                            for isig in range(int(nb_cplx_per_samples/2)):
                                comp_sig.update([('comp_sig_%i'%int(isig), 
                                                  temp[:, 1 + 2 * (isig) - 1 ] + complex(0,1) * temp[:, 2+2*(isig)-1])  ]  )
                            head.update({'comp_sig':comp_sig})
                            self.cmpPwrEK80(ping = head, config= self.config_transceiver)
                    
                    self.pings[head['cid']][ping_n] = head  
        #get power, y and complex np arrays in shape 

        self.power_data  = [defaultdict] *  self.n_trans
        self.y_data  = [defaultdict] *  self.n_trans
        self.comp_sig_data  = [defaultdict] *  self.n_trans
        
        for c in range(self.n_trans):
            if 'power' in self.pings[c][0]:
                self.power_data[c] = np.array([self.pings[c][x]['power'] for x in self.pings[c]]).squeeze()
            if 'y' in self.pings[c][0]:                    
                self.y_data[c] = np.array([self.pings[c][x]['y'] for x in self.pings[c]]).squeeze()
            if 'comp_sig' in self.pings[c][0]:
                self.comp_sig_data[c] = np.array([np.transpose(np.asarray([(v) for v in self.pings[c][x]['comp_sig'].values()])) for x in self.pings[c]]).squeeze()

        params = pd.DataFrame(self.parameters).transpose()
        self.params = params.drop_duplicates()
