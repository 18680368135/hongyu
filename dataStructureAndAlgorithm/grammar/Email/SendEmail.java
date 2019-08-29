package Email;



import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.AddressException;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;
import java.util.Properties;

public class SendEmail {

    public static void main(String args[]) throws AddressException, MessagingException{

        //收件人电子邮箱
        String to = "abcd@gmail.com";
        String from = "web@gmail.com";

        //指定发送邮件的主机为localhost
        String host = "localhost";

        //获取系统属性
        Properties properties= System.getProperties();

        //设置邮件服务器
        properties.setProperty("mail.smtp.host", host);
        // properties.setProperty("mail.smtp.port", "8080");

        //获取默认session对象
        Session session= Session.getDefaultInstance(properties);

        //创建默认的MimeMessage对象
        MimeMessage message = new MimeMessage(session);

        // Set From : 头部头字段
        message.setFrom(new InternetAddress(from));

        //Set to :头部头字段
        message.addRecipient(Message.RecipientType.TO, new InternetAddress(to));

        //set subject: 头部头字段
        message.setSubject("this is the Subject line");

        // 设置消息体
        message.setText("this is actual message");

        //发送消息
        Transport.send(message);
        System.out.println("Sent Message successfully .........");




    }
}
