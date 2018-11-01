# Install python & some tips
First, check wheater you have installed python and what version it is.<br/>
Second, according to your operating system and your own need to choose which version to install and the way to install.<br/>
Sometimes, you shoulde be careful about knowing the link *python* *python3* *python2* and so on that which version they are really pointing.<br/>
# Install pip
sudo pacman -S python-pip
# change mirrors
<pre>
pip install pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
</pre>
