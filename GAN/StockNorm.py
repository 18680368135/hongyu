import cmath
import stockstats
class StockNorm(object):
    def __init__(self, close, close_inter, inter):
        self.close = close
        self.close_inter = close_inter
        self.interval = inter

    def bias(self, MA):
        """
        计算乖离率bias,
        N日的乖离率=(当日收盘价－N日移动平均价)/N日移 动平均价×100％
        bias(N)=[CN－MA(N)]/MA(N)×100％

        function:乖离率bias指标的一般研判标准主要集中在乖离率正负值转换和乖离率取值等方面研判上。
        1.2.1.2、 乖离率正负值转换 乖离率有正乖离率和负乖离率之分。
        1.2.1.2、若股价在移动平均线之上，则为正乖离率；股价在移动平均线之下，则为负乖离率；当股价与移动平均线相交，则乖离率为零。
        正的乖离率越大，表明短期股价涨幅过大，随之而来的是多头的短线获利颇丰，因此，股价再度上涨的压力加大，股价可能受短线
        获利盘的打压而下跌的可能能越高。反之，负的乖离率越大，空头回补而使股价反弹的可能越大。
        2、在投机性很强的市道和个股上，市场的投机性越高，乖离率的弹性越大，个别股票的乖离率差异更大，随股性而变化。
        :return:bias
        """
        bias = (self.close - MA) / MA
        bias = round(bias, 3)
        return bias


    def movingAverages(self,):
        """
        MA = (X1+X2+X3+…..+Xn)/N
        X1 .... XN 为N天的收盘价
        :param close_inter: inter天的收盘价
        :return:
        """
        MA = sum(self.close_inter)/self.interval
        return MA


    def exponentialMovingAverages(self):
        ema = 0
        N = self.interval
        n = 1
        for ci in self.close_inter[-N:]:
            ema = (2 * ci + (n - 1) * ema) / (n + 1)
            n += 1
        return ema

    def bollingerBand(self,close_inter, MA):
        """
        1.2.1.2.日BOLL指标的计算公式
           中轨线=N日的移动平均线
           上轨线=中轨线+两倍的标准差
           下轨线=中轨线－两倍的标准差
        2.日BOLL指标的计算过程
           （1.2.1.2）计算MA
                 MA=N日内的收盘价之和÷N
           （2）计算标准差MD
                 MD=平方根(N日的（C－MA）的两次方之和除以N)
           （3）计算MB、UP、DN线
                 MB=（N－1.2.1.2）日的MA
                 UP=MB+2×MD
                 DN=MB－2×MD
        :param close_inter: N天收盘价
        :param MA: N天的均价
        :return:
        """
        N = self.interval
        for i in range(N):
            close_inter[i] = close_inter[i] - MA
            close_inter[i] *= close_inter[i]
        MD = cmath.sqrt(sum(close_inter)/N)
        close_inter_pos = close_inter.remove(close_inter[0])
        MB = sum(close_inter_pos)/(N-1)
        UP = MB + 2 * MD
        DN = MB - 2 * MD

        return UP, DN

    def movingAverageConvergenceDivergence(self, close):
        """
        异同移动平均线，是从双指数移动平均线发展而来的，由快的指数移动平均线（EMA12）
        减去慢的指数移动平均线（EMA26）得到快线DIF，
        再用2×（快线DIF-DIF的9日加权移动均线DEA）得到MACD柱

        计算过程
        1.2.1.2、计算移动平均值（EMA）
            12日EMA的算式为
            EMA（12）=前一日EMA（12）×11/13+今日收盘价×2/13
            26日EMA的算式为
            EMA（26）=前一日EMA（26）×25/27+今日收盘价×2/27
        2、计算离差值（DIF）
            DIF=今日EMA（12）－今日EMA（26）
        3、计算DIF的9日EMA
           根据离差值计算其9日的EMA，即离差平均值，是所求的MACD值。为了不与指标原名相混淆，此值又名DEA或DEM。
           今日DEA（MACD）=前一日DEA×8/10+今日DIF×2/10

           计算出的DIF和DEA的数值均为正值或负值。
           用（DIF-DEA）×2即为MACD柱状图。
        :param close:
        :return: MACD
        """

    def directionalMovementIndex(self):
        """
        动向指数也叫平均方向指标
        :return:
        """


