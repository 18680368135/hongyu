package serializable;



import java.io.Serializable;

class Rectangle implements Serializable {

    private static final long serialVersionUID = 178456561841188419L;
    private Integer width;
    private Integer height;
    private transient Integer area;


    public Rectangle(Integer width, Integer height){
        this.width = width;
        this.height = height;
        this.area = width * height;
    }

    public void setArea(){
        this.area = this.width * this.height;
    }


    @Override
    public String toString(){
        StringBuffer sb= new StringBuffer(40);
        sb.append("width :");
        sb.append(this.width);
        sb.append(" height :");
        sb.append(this.height);
        sb.append("\narea :");
        sb.append(this.area);
        return sb.toString();
    }
}

